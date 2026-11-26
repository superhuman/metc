"""Run exact sequential JP-ERRANT workers in ordered process-level shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-orig", required=True)
    parser.add_argument("-cor", nargs="+", required=True)
    parser.add_argument("-out", required=True)
    parser.add_argument("-lang", default="en")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if len(args.cor) != 1:
        parser.error("the paper adapter requires exactly one correction per source")
    if args.workers < 1:
        parser.error("--workers must be positive")

    sources = Path(args.orig).read_text(encoding="utf-8").splitlines()
    corrections = Path(args.cor[0]).read_text(encoding="utf-8").splitlines()
    if len(sources) != len(corrections):
        raise RuntimeError(
            f"source/correction line counts differ: {len(sources)} != {len(corrections)}"
        )
    if not sources:
        Path(args.out).write_text("", encoding="utf-8")
        return 0
    shard_count = min(args.workers, len(sources))
    shard_size = (len(sources) + shard_count - 1) // shard_count

    with tempfile.TemporaryDirectory(prefix="jp-errant-shards-") as temporary:
        root = Path(temporary)
        commands: list[tuple[str, ...]] = []
        outputs: list[Path] = []
        for shard in range(shard_count):
            start = shard * shard_size
            end = min(start + shard_size, len(sources))
            if start == end:
                continue
            source_path = root / f"source-{shard}.txt"
            correction_path = root / f"correction-{shard}.txt"
            output_path = root / f"output-{shard}.m2"
            source_path.write_text(
                "".join(value + "\n" for value in sources[start:end]),
                encoding="utf-8",
            )
            correction_path.write_text(
                "".join(value + "\n" for value in corrections[start:end]),
                encoding="utf-8",
            )
            commands.append(
                (
                    sys.executable,
                    args.worker,
                    "--mode",
                    "sequential",
                    "-orig",
                    str(source_path),
                    "-cor",
                    str(correction_path),
                    "-out",
                    str(output_path),
                    "-lang",
                    args.lang,
                )
            )
            outputs.append(output_path)

        def execute(command: tuple[str, ...]) -> None:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"JP-ERRANT shard exited with {completed.returncode}: "
                    f"{completed.stderr.strip()}"
                )

        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            list(executor.map(execute, commands))

        with Path(args.out).open("w", encoding="utf-8", newline="\n") as destination:
            for output_path in outputs:
                destination.write(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
