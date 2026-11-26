"""Sequential-exact JP-ERRANT worker for the pinned external environment."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Sequence


def _chunks(values: Sequence[str], size: int) -> list[Sequence[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-orig", required=True)
    parser.add_argument("-cor", nargs="+", required=True)
    parser.add_argument("-out", required=True)
    parser.add_argument("-lang", default="en")
    parser.add_argument("--mode", choices=("sequential", "bulk"), default="sequential")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if len(args.cor) != 1:
        parser.error("the paper adapter requires exactly one correction per source")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    # Avoid shadowing the external jp_errant package with the sibling
    # orchestrator module when this worker is executed as a file.
    worker_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != worker_directory
    ]

    import stanza  # type: ignore[import-not-found]
    from jp_errant.annotator import Annotator  # type: ignore[import-not-found]

    resources = os.environ.get("STANZA_RESOURCES_DIR")
    if not resources:
        raise RuntimeError("STANZA_RESOURCES_DIR is required")
    pipeline_options = {
        "dir": resources,
        "download_method": stanza.DownloadMethod.REUSE_RESOURCES,
        "use_gpu": args.use_gpu,
        "verbose": False,
    }
    if args.lang in {"en", "de", "uk", "cs"}:
        pipeline = stanza.Pipeline(
            args.lang,
            processors="tokenize,pos,mwt,lemma,depparse",
            **pipeline_options,
        )
    elif args.lang == "zh":
        models = os.environ.get("ZH_GSDLTP_DIR")
        if not models:
            raise RuntimeError("ZH_GSDLTP_DIR is required for Chinese")
        pipeline = stanza.Pipeline(
            lang="zh-hans",
            processors="tokenize,pos",
            tokenize_model_path=f"{models}/tokenize/zh-hans_gsdsimp_tokenizer.pt",
            pos_model_path=f"{models}/pos/zh-hans_gsdsimp_charlm_tagger.pt",
            **pipeline_options,
        )
    else:
        pipeline = stanza.Pipeline(args.lang, **pipeline_options)

    merger = import_module(f"jp_errant.{args.lang}.merger")
    classifier = import_module(f"jp_errant.{args.lang}.classifier")
    classifier.nlp = pipeline
    annotator = Annotator(args.lang, pipeline, merger, classifier)

    sources = Path(args.orig).read_text(encoding="utf-8").splitlines()
    corrections = Path(args.cor[0]).read_text(encoding="utf-8").splitlines()
    if len(sources) != len(corrections):
        raise RuntimeError(
            f"source/correction line counts differ: {len(sources)} != {len(corrections)}"
        )

    def write_sentence(output, source, correction, original_doc, correction_doc=None):
        """Mirror upstream parallel_to_m2's one-row serialization."""

        source_tokens = " ".join(word.text for word in original_doc.iter_words())
        output.write(f"S {source_tokens}\n")
        if original_doc.text.replace(" ", "").strip() == correction.replace(
            " ", ""
        ).strip():
            output.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
        else:
            if correction_doc is None:
                correction_doc = pipeline(correction.strip())
            for edit in annotator.annotate(
                original_doc,
                correction_doc,
                lev=False,
                merging="rules",
            ):
                output.write(edit.to_m2(0) + "\n")
        output.write("\n")

    with Path(args.out).open("w", encoding="utf-8", newline="\n") as output:
        if args.mode == "sequential":
            for source, correction in zip(sources, corrections, strict=True):
                write_sentence(
                    output,
                    source,
                    correction,
                    pipeline(source.strip()),
                )
        else:
            documents = []
            texts = [*sources, *corrections]
            for chunk in _chunks(texts, args.batch_size):
                documents.extend(pipeline.bulk_process(list(chunk)))
            for source, correction, original_doc, correction_doc in zip(
                sources,
                corrections,
                documents[: len(sources)],
                documents[len(sources) :],
                strict=True,
            ):
                write_sentence(output, source, correction, original_doc, correction_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
