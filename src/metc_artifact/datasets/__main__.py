"""Command-line entry point for public benchmark reconstruction."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .catalog import RECIPES, SOURCES
from .registry import (
    assert_semantically_equal,
    read_jsonl,
    reconstruct_all_from_public_sources,
    reconstruct_from_public_sources,
    write_jsonl,
)


def _catalog(_args: argparse.Namespace) -> int:
    payload = {
        "sources": {key: dataclasses.asdict(value) for key, value in SOURCES.items()},
        "datasets": {key: dataclasses.asdict(value) for key, value in RECIPES.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _reconstruct(args: argparse.Namespace) -> int:
    rows = reconstruct_from_public_sources(
        args.dataset,
        args.cache_dir,
        cherrant_python=args.cherrant_python,
        no_network=args.no_network,
    )
    write_jsonl(rows, args.output)
    print(f"wrote {len(rows)} records to {args.output}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    rows = reconstruct_from_public_sources(
        args.dataset,
        args.cache_dir,
        cherrant_python=args.cherrant_python,
        no_network=args.no_network,
    )
    assert_semantically_equal(args.dataset, rows, read_jsonl(args.frozen))
    print(f"{args.dataset}: public reconstruction matches {args.frozen}")
    return 0


def _reconstruct_all(args: argparse.Namespace) -> int:
    outputs = reconstruct_all_from_public_sources(
        args.cache_dir,
        args.output_dir,
        cherrant_python=args.cherrant_python,
        no_network=args.no_network,
    )
    for dataset_id, output in outputs.items():
        print(f"{dataset_id}: {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m metc_artifact.datasets",
        description="Acquire public corpora and reconstruct METC paper references.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="print source pins, terms, and expected counts")
    catalog.set_defaults(handler=_catalog)

    reconstruct_all = commands.add_parser(
        "reconstruct-all", help="reconstruct all seven benchmark references"
    )
    reconstruct_all.add_argument("--cache-dir", type=Path, required=True)
    reconstruct_all.add_argument("--output-dir", type=Path, required=True)
    reconstruct_all.add_argument("--cherrant-python", type=Path, required=True)
    reconstruct_all.add_argument("--no-network", action="store_true")
    reconstruct_all.set_defaults(handler=_reconstruct_all)

    for name, handler in (("reconstruct", _reconstruct), ("verify", _verify)):
        command = commands.add_parser(name)
        command.add_argument("dataset", choices=tuple(RECIPES))
        command.add_argument("--cache-dir", type=Path, required=True)
        command.add_argument("--cherrant-python", type=Path)
        command.add_argument("--no-network", action="store_true")
        if name == "reconstruct":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--frozen", type=Path, required=True)
        command.set_defaults(handler=handler)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
