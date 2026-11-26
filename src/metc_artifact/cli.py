"""Commands for complete METC artifact verification."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .data_artifact import HUMAN_STUDY_DATASETS, DataArtifactError, validate_data_artifact
from .metrics import compute_human_tables, compute_paper_tables, load_human_study, load_judge_rows
from .reference_matching import score_reference_matching
from .schema import DATASETS, dataset_spec
from .semantic import compare_annotations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_data(args: argparse.Namespace) -> int:
    try:
        report = validate_data_artifact(args.data_root)
    except DataArtifactError as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    spec = dataset_spec(args.dataset)
    comparison = compare_annotations(
        _read_jsonl(args.expected),
        _read_jsonl(args.actual),
        spec,
        systems=args.systems,
        max_differences=args.max_differences,
    )
    print(json.dumps(comparison.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if comparison.equal else 1


def _reference(args: argparse.Namespace) -> int:
    selected = (
        tuple(dataset_spec(identifier) for identifier in args.datasets)
        if args.datasets
        else DATASETS
    )
    seen: set[str] = set()
    for spec in selected:
        if spec.dataset_id in seen:
            continue
        seen.add(spec.dataset_id)
        source = (
            args.data_root
            / "benchmarks"
            / spec.dataset_id
            / "annotations.jsonl"
        )
        rows = _read_jsonl(source)
        table = score_reference_matching(
            rows,
            spec,
            annotation_source=args.annotation_source,
            span_match=args.span_match,
            samples=args.samples,
            seed=args.seed,
        )
        destination = args.output_dir / f"{spec.dataset_id}.csv"
        _write_csv(destination, table)
        print(f"{spec.dataset_id}\t{destination}")
    return 0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _coverage_error(kind: str, expected: set[str], observed: set[str]) -> int:
    """Report incomplete or unexpected inputs without writing any table."""

    print(
        json.dumps(
            {
                "error": f"{kind} do not cover exactly the released datasets",
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


def _judge_tables(args: argparse.Namespace) -> int:
    """Recompute the paper's judge tables from the released frozen verdicts."""

    corpus = load_judge_rows(args.data_root, judge_source="frozen_authoritative")
    expected = {spec.dataset_id for spec in DATASETS}
    observed = {str(row.get("dataset")) for row in corpus.rows}
    if observed != expected:
        return _coverage_error("judge verdicts", expected, observed)
    tables = compute_paper_tables(corpus, samples=args.samples, seed=args.seed)
    destination = args.output_dir / "frozen-authoritative-tables.json"
    _write_json(destination, tables)
    print(f"judge_tables\t{destination}")
    return 0


def _human_tables(args: argparse.Namespace) -> int:
    """Recompute the human-study tables from the released assignment records."""

    manifest = json.loads((args.data_root / "manifest.full.json").read_text(encoding="utf-8"))
    studies = manifest.get("human_annotations")
    if not isinstance(studies, dict):
        raise ValueError("manifest.full.json lacks human_annotations")
    if set(studies) != HUMAN_STUDY_DATASETS:
        return _coverage_error("human studies", set(HUMAN_STUDY_DATASETS), set(studies))
    # Output filenames come only from the fixed, validated study inventory.
    for dataset_id in sorted(HUMAN_STUDY_DATASETS):
        corpus = load_human_study(args.data_root, dataset_id)
        tables = compute_human_tables(corpus, samples=args.samples, seed=args.seed)
        destination = args.output_dir / f"{dataset_id}.json"
        _write_json(destination, tables)
        print(f"{dataset_id}\t{destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metc-artifact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-data", help="validate the complete extracted Data artifact"
    )
    verify.add_argument("data_root", type=Path)
    verify.set_defaults(handler=_verify_data)

    compare = subparsers.add_parser(
        "compare-annotations",
        help="semantically compare regenerated and authoritative annotations",
    )
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--systems", nargs="+")
    compare.add_argument("--max-differences", type=int, default=100)
    compare.set_defaults(handler=_compare)

    reference = subparsers.add_parser(
        "reference-matching",
        help="recompute reference matching from complete or fresh annotations",
    )
    reference.add_argument("--data-root", type=Path, required=True)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument("--datasets", nargs="+")
    reference.add_argument(
        "--annotation-source",
        choices=("authoritative_frozen", "fresh_rerun"),
        required=True,
        help="identify whether the supplied annotations are frozen or freshly rerun",
    )
    reference.add_argument("--span-match", choices=("overlap", "exact"), default="overlap")
    reference.add_argument("--samples", type=int, default=1000)
    reference.add_argument("--seed", type=int, default=42)
    reference.set_defaults(handler=_reference)

    judge = subparsers.add_parser(
        "judge-tables",
        help="recompute the paper judge tables from the frozen released verdicts (no API calls)",
    )
    judge.add_argument("--data-root", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--samples", type=int, default=1000)
    judge.add_argument("--seed", type=int, default=42)
    judge.set_defaults(handler=_judge_tables)

    human = subparsers.add_parser(
        "human-tables",
        help="recompute the human-evaluation tables from the released assignments",
    )
    human.add_argument("--data-root", type=Path, required=True)
    human.add_argument("--output-dir", type=Path, required=True)
    human.add_argument("--samples", type=int, default=10_000)
    human.add_argument("--seed", type=int, default=42)
    human.set_defaults(handler=_human_tables)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
