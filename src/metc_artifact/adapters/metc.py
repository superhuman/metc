"""METC batch adapter: canonical pairs -> normalized fresh annotations."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec

from .contract import AnnotationRecord, NormalizedEdit, PairRecord, read_pairs, write_annotations
from .errors import AdapterInputError, AdapterOutputError
from .m2 import token_char_spans
from .mappings import map_metc
from .process import require_directory, require_executable, run_checked


def _utf16_to_codepoint(text: str, offset: int) -> int:
    if offset < 0:
        raise AdapterOutputError(f"negative UTF-16 offset: {offset}")
    consumed = 0
    for index, character in enumerate(text):
        if consumed == offset:
            return index
        width = 2 if ord(character) > 0xFFFF else 1
        if consumed < offset < consumed + width:
            raise AdapterOutputError(f"UTF-16 offset {offset} splits a surrogate pair")
        consumed += width
    if consumed == offset:
        return len(text)
    raise AdapterOutputError(f"UTF-16 offset {offset} exceeds source length {consumed}")


def _char_to_token(
    pair: PairRecord, start: int, end: int
) -> tuple[int, int]:
    spans = token_char_spans(pair.source_tokens or (), " ")
    if start == end:
        for index, (token_start, _token_end) in enumerate(spans):
            if token_start >= start:
                return index, index
        return len(spans), len(spans)
    selected = [
        index
        for index, (token_start, token_end) in enumerate(spans)
        if token_end > start and token_start < end
    ]
    if not selected:
        boundary = next(
            (
                index
                for index, (token_start, _token_end) in enumerate(spans)
                if token_start >= start
            ),
            len(spans),
        )
        return boundary, boundary
    return selected[0], selected[-1] + 1


def _token_span_replacement(
    pair: PairRecord,
    token_start: int,
    token_end: int,
    char_start: int,
    char_end: int,
    char_replacement: str,
) -> str:
    """Expand a character edit to the complete covered corpus-token span."""

    if token_start == token_end:
        return char_replacement
    spans = token_char_spans(pair.source_tokens or (), " ")
    token_char_start = spans[token_start][0]
    token_char_end = spans[token_end - 1][1]
    return (
        pair.source[token_char_start:char_start]
        + char_replacement
        + pair.source[char_end:token_char_end]
    )


def _read_cli_output(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterOutputError(
                    f"METC output line {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise AdapterOutputError(f"METC output line {line_number} is not an object")
            output.append(value)
    return output


def _integer_field(raw: Mapping[str, Any], field: str, pair: PairRecord) -> int:
    value = raw.get(field)
    if type(value) is not int:
        raise AdapterOutputError(
            f"{pair.item_id}: METC field {field!r} must be an exact integer"
        )
    return value


def _normalize_annotation(
    pair: PairRecord,
    spec: DatasetSpec,
    raw: Mapping[str, Any],
    *,
    tokenized: bool,
) -> NormalizedEdit:
    label = raw.get("label")
    sublabel = raw.get("sublabel")
    if not isinstance(label, str) or not isinstance(sublabel, str):
        raise AdapterOutputError(f"{pair.item_id}: METC annotation has no label/sublabel")
    native_label = f"{label}/{sublabel}"
    replacement = raw.get("replacement")
    if replacement is None:
        replacement = ""
    elif not isinstance(replacement, str):
        raise AdapterOutputError(f"{pair.item_id}: METC replacement must be a string")
    if tokenized:
        start = _integer_field(raw, "startToken", pair)
        end = _integer_field(raw, "endToken", pair)
        return NormalizedEdit(
            start=start,
            end=end,
            replacement=replacement,
            native_label=native_label,
            mapped_label=map_metc(spec.dataset_id, native_label),
        )

    utf16_start = _integer_field(raw, "start", pair)
    utf16_end = _integer_field(raw, "end", pair)
    char_start = _utf16_to_codepoint(pair.source, utf16_start)
    char_end = _utf16_to_codepoint(pair.source, utf16_end)
    if spec.edit_format == "token_dict":
        start, end = _char_to_token(pair, char_start, char_end)
        token_replacement = _token_span_replacement(
            pair,
            start,
            end,
            char_start,
            char_end,
            replacement,
        )
        return NormalizedEdit(
            start=start,
            end=end,
            replacement=token_replacement,
            native_label=native_label,
            mapped_label=map_metc(spec.dataset_id, native_label),
            char_start=char_start,
            char_end=char_end,
            char_source=pair.source[char_start:char_end],
            char_replacement=replacement,
        )
    return NormalizedEdit(
        start=char_start,
        end=char_end,
        replacement=replacement,
        native_label=native_label,
        mapped_label=map_metc(spec.dataset_id, native_label),
    )


def run_metc(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    checkpoint: str | Path,
    batch_size: int = 8,
    device: str = "cuda",
    inference_profile: str = "paper",
    allow_nonpaper: bool = False,
    environment: Mapping[str, str] | None = None,
) -> tuple[AnnotationRecord, ...]:
    if inference_profile != "paper" and not allow_nonpaper:
        raise AdapterInputError(
            "portable METC inference is diagnostic only and is known not to reproduce the "
            "frozen paper annotations; pass allow_nonpaper=True explicitly"
        )
    if inference_profile == "paper" and device != "cuda":
        raise AdapterInputError(
            "the exact paper profile requires CUDA bfloat16, FlashAttention 2, "
            "fixed 512-token inference, and reference_compile"
        )
    executable = require_executable(python, "METC isolated Python executable")
    model = require_directory(checkpoint, "hydrated METC checkpoint")
    for pair in pairs:
        pair.validate_for(spec)
    if not pairs:
        return ()
    tokenized = spec.dataset_id == "uk_uagec"
    with tempfile.TemporaryDirectory(prefix="metc-adapter-") as temporary:
        root = Path(temporary)
        cli_input = root / "pairs.jsonl"
        cli_output = root / "predictions.jsonl"
        with cli_input.open("w", encoding="utf-8", newline="\n") as stream:
            for pair in pairs:
                stream.write(
                    json.dumps(
                        {
                            "source": pair.source,
                            "target": pair.correction,
                            "tokenized": tokenized,
                            "cleanup_semantic": False,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        run_checked(
            (
                executable,
                "-m",
                "metc.cli",
                "classify",
                "--model",
                model,
                "--input",
                cli_input,
                "--output",
                cli_output,
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--inference-profile",
                inference_profile,
            ),
            environment=environment,
        )
        raw_outputs = _read_cli_output(cli_output)
    if len(raw_outputs) != len(pairs):
        raise AdapterOutputError(
            f"METC returned {len(raw_outputs)} rows for {len(pairs)} pairs"
        )
    records: list[AnnotationRecord] = []
    for pair, raw_output in zip(pairs, raw_outputs, strict=True):
        raw_annotations = raw_output.get("annotations")
        if not isinstance(raw_annotations, list):
            raise AdapterOutputError(f"{pair.item_id}: METC output has no annotation array")
        record = AnnotationRecord(
            dataset_id=spec.dataset_id,
            item_id=pair.item_id,
            system="metc",
            span_unit=spec.evaluation_span_unit,
            edits=tuple(
                _normalize_annotation(pair, spec, raw, tokenized=tokenized)
                for raw in raw_annotations
            ),
        )
        record.validate_for_pair(pair, spec)
        records.append(record)
    return tuple(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-profile", default="paper")
    parser.add_argument("--allow-nonpaper", action="store_true")
    args = parser.parse_args(argv)
    import os

    spec = dataset_spec(args.dataset)
    records = run_metc(
        read_pairs(args.pairs, spec),
        spec,
        python=args.python or os.environ.get("METC_SYSTEM_PYTHON"),
        checkpoint=args.checkpoint or os.environ.get("METC_MODEL_PATH"),
        batch_size=args.batch_size,
        device=args.device,
        inference_profile=args.inference_profile,
        allow_nonpaper=args.allow_nonpaper,
    )
    write_annotations(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
