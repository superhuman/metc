"""English ERRANT adapter with the NLP import confined to a child process."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec

from .contract import AnnotationRecord, NormalizedEdit, PairRecord, read_pairs, write_annotations, write_pairs
from .errors import AdapterOutputError
from .mappings import map_errant_en
from .process import require_executable, require_file, run_checked


DEFAULT_WORKER = Path(__file__).with_name("_errant_worker.py")


def run_errant(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    worker: str | Path = DEFAULT_WORKER,
    environment: Mapping[str, str] | None = None,
) -> tuple[AnnotationRecord, ...]:
    if spec.dataset_id not in {"en_conll14_ann0", "en_conll14_ann1"}:
        raise AdapterOutputError(f"ERRANT does not apply to {spec.dataset_id}")
    executable = require_executable(python, "ERRANT isolated Python executable")
    worker_path = require_file(worker, "ERRANT isolated worker")
    for pair in pairs:
        pair.validate_for(spec)
    if not pairs:
        return ()
    with tempfile.TemporaryDirectory(prefix="errant-adapter-") as temporary:
        root = Path(temporary)
        worker_input = root / "pairs.jsonl"
        worker_output = root / "raw.jsonl"
        write_pairs(worker_input, pairs)
        run_checked(
            (executable, worker_path, "--input", worker_input, "--output", worker_output),
            environment=environment,
        )
        rows = [
            json.loads(line)
            for line in worker_output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if len(rows) != len(pairs):
        raise AdapterOutputError(f"ERRANT returned {len(rows)} rows for {len(pairs)} pairs")
    records: list[AnnotationRecord] = []
    for pair, row in zip(pairs, rows, strict=True):
        if row.get("item_id") != pair.item_id or not isinstance(row.get("edits"), list):
            raise AdapterOutputError(f"ERRANT output misaligned at {pair.item_id}")
        edits = []
        for raw in row["edits"]:
            native_label = raw.get("native_label")
            replacement = raw.get("replacement")
            if not isinstance(native_label, str) or not isinstance(replacement, str):
                raise AdapterOutputError(
                    f"{pair.item_id}: ERRANT label/replacement must be strings"
                )
            start = raw.get("start")
            end = raw.get("end")
            if type(start) is not int or type(end) is not int:
                raise AdapterOutputError(
                    f"{pair.item_id}: ERRANT span boundaries must be exact integers"
                )
            edits.append(
                NormalizedEdit(
                    start=start,
                    end=end,
                    replacement=replacement,
                    native_label=native_label,
                    mapped_label=map_errant_en(native_label),
                )
            )
        record = AnnotationRecord(
            dataset_id=spec.dataset_id,
            item_id=pair.item_id,
            system="errant",
            span_unit=spec.evaluation_span_unit,
            edits=tuple(edits),
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
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    args = parser.parse_args(argv)
    spec = dataset_spec(args.dataset)
    records = run_errant(
        read_pairs(args.pairs, spec),
        spec,
        python=args.python or os.environ.get("ERRANT_PYTHON"),
        worker=args.worker,
    )
    write_annotations(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
