#!/usr/bin/env python3
"""Download only the Stanza 1.10.0 model set executed by the paper adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import stanza


LANGUAGES = ("cs", "de", "en", "ko", "uk", "zh-hans")
RESOURCES_URL = "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main"
RESOURCES_VERSION = "1.10.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        stanza.download(
            language,
            model_dir=str(args.output),
            resources_url=RESOURCES_URL,
            resources_version=RESOURCES_VERSION,
            verbose=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
