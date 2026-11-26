"""Executed only inside the pinned ERRANT environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Executing this file directly places the adapters directory first on
    # ``sys.path``.  Remove it before importing the external package so the
    # sibling ``errant.py`` orchestrator cannot shadow official ERRANT.
    worker_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != worker_directory
    ]
    import errant  # type: ignore[import-not-found]  # isolated external environment

    annotator = errant.load("en")
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            original = row["source"]
            correction = row["correction"]
            original_doc = annotator.parse(original)
            correction_doc = annotator.parse(correction)
            edits = []
            for edit in annotator.annotate(original_doc, correction_doc):
                if edit.type in {None, "noop", "NA"}:
                    continue
                start = (
                    original_doc[edit.o_start].idx
                    if edit.o_start < len(original_doc)
                    else len(original)
                )
                end = (
                    original_doc[edit.o_end - 1].idx
                    + len(original_doc[edit.o_end - 1].text)
                    if edit.o_end > edit.o_start
                    else start
                )
                edits.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "replacement": edit.c_str or "",
                        "native_label": str(edit.type),
                    }
                )
            destination.write(
                json.dumps(
                    {"item_id": row["item_id"], "edits": edits},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
