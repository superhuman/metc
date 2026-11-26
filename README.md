# METC: Multilingual Error Type Classifier

METC labels the edits in a grammatical error correction. Given a source sentence
and its correction, it extracts the edits, assigns each edit one of 42 shared
error-type labels, and returns character offsets. One checkpoint and one
taxonomy cover all languages. There are no language-specific linguistic
analyzers or classification heads. The edit extractor does use fixed
script-aware segmentation rules (see `TOKENIZER_MODIFICATIONS.md` and
`src/metc/extractor/`).

This repository contains the code, model metadata, prompts, taxonomy, and
evaluation pipeline for the paper *METC: Multilingual Error Type Classifier for
Grammatical Error Correction*. The paper will appear in *Findings of the
Association for Computational Linguistics: EACL 2027*. A link to the published
version and the final BibTeX entry will be added here once it is available.

The model checkpoint and the evaluation data are published as assets of the
[`v1.0.0` release](https://github.com/superhuman/metc/releases/tag/v1.0.0).

```text
She go to school.  ->  She went to school.
She {go=>went:::Grammar/VerbTenseForm} to school.
```

## Contents

| Path | What it is |
|---|---|
| `src/metc/` | METC inference: edit extractor, classifier, label mapping, `metc` CLI |
| `src/metc_artifact/` | Paper evaluation: dataset reconstruction, baseline adapters, LLM judges, metrics, `metc-artifact` CLI |
| `reproduce.py` | One-command end-to-end reproduction of the paper evaluation |
| `prompts/` | Teacher annotation prompt, LLM judge prompt, and the 42-label taxonomy (`taxonomy_v2.md`) |
| `evaluation/` | Cross-taxonomy label mappings and the plain-language label glosses |
| `human_evaluation/` | Annotation guidelines used for the human evaluation |
| `model/` | Checkpoint contract (`checkpoint.json`) and model notices |
| `training/` | Released training configuration and aggregate training-corpus distribution |
| `baselines/gerrant_de/` | German ERRANT (gERRANT) port used as a baseline |
| `MODEL_CARD.md` | Model details, training, intended use, and limitations |

Release assets (not in git):

| Asset | Size | SHA-256 |
|---|---:|---|
| `metc-checkpoint.zip` | 521,562,261 B | `06ddc2bc1425187bc358785fc3af5febba910ee8c5a38e8b2b7be44cf905f8a3` |
| `metc-data.zip` | 37,828,516 B | `c47a99032b8ecec17d553439a6b9a5045600b16221efae12dbafa00602a914ab` |

## Quickstart: run METC

Requirements: CPython 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone --branch v1.0.0 https://github.com/superhuman/metc.git
cd metc

curl --fail -L -O https://github.com/superhuman/metc/releases/download/v1.0.0/metc-checkpoint.zip \
  && echo "06ddc2bc1425187bc358785fc3af5febba910ee8c5a38e8b2b7be44cf905f8a3  metc-checkpoint.zip" | sha256sum -c \
  && unzip metc-checkpoint.zip
```

`--branch v1.0.0` checks out the release the assets belong to. Each step runs
only if the previous one succeeded, so a failed download or a hash mismatch
stops before anything is extracted. On macOS, use `shasum -a 256 -c` instead of
`sha256sum -c`.

The checkpoint archive contains the four model files but not the tokenizer
files. Prepare the exact tokenizer once:

```bash
uv run --frozen metc prepare-tokenizer --model metc-checkpoint
```

This downloads four files from `jhu-clsp/mmBERT-small` revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`, checks their sizes and SHA-256
hashes, adds `[EDIT]` and `[/EDIT]` as token IDs 256000 and 256001, and checks
the three generated tokenizer files. After this step, inference runs offline.

Classify one pair:

```bash
uv run --frozen metc classify --model metc-checkpoint \
  --source "She go to school." --target "She went to school."
```

```json
{"annotations": [{"end": 6, "label": "Grammar", "replacement": "went", "start": 4, "sublabel": "VerbTenseForm"}], "classified": "She {go=>went:::Grammar/VerbTenseForm} to school."}
```

Or stream a JSONL file of `{"source": ..., "target": ...}` objects:

```bash
uv run --frozen metc classify --model metc-checkpoint \
  --input pairs.jsonl --output predictions.jsonl --batch-size 8
```

`start` and `end` are offsets into the source sentence, counted in UTF-16 code
units. Labels are `label/sublabel` pairs from the taxonomy in
`prompts/taxonomy_v2.md`.

### Inference profiles

The default `portable` profile runs in fp32 on CPU, MPS, or CUDA. It is meant
for general use and diagnostics: a small number of near-tie labels can differ
from the evaluated service, and because adjacent edits are merged by label,
this can also change a few merged spans. The `paper` profile reproduces the evaluated serving
setup exactly. It needs Linux, an NVIDIA GPU with bfloat16 support, CUDA 12.8,
and the `paper` extra (FlashAttention 2):

```bash
uv run --frozen --extra paper metc classify --model metc-checkpoint \
  --device cuda --inference-profile paper \
  --source "She go to school." --target "She went to school."
```

`model/checkpoint.json` pins the archive, every runtime file, the 42-label
architecture, and the paper inference profile. `metc inspect-checkpoint` prints
the active contract.

Inputs are bounded to 3,584 Unicode characters and then truncated on the right
to 512 tokens. If truncation removes trailing edit markers, those edits receive
the `Other/Compound_Complex` fallback label used by the evaluated service. A
fallback prediction has no defined confidence: the low-level API reports `0.0`
as a sentinel, and evaluation ignores it.

## Reproduce the paper

The paper evaluation can be checked at three levels. They answer different
questions, so they are kept separate.

Download and check the evaluation data first:

```bash
curl --fail -L -O https://github.com/superhuman/metc/releases/download/v1.0.0/metc-data.zip \
  && echo "c47a99032b8ecec17d553439a6b9a5045600b16221efae12dbafa00602a914ab  metc-data.zip" | sha256sum -c \
  && unzip metc-data.zip \
  && uv run --frozen metc-artifact verify-data metc-data
```

`metc-data.zip` contains the benchmark sources and corrections, public
references, every system's annotations, the de-identified human annotations and
their sampling frames, the teacher outputs, and both frozen LLM-judge runs. It
has its own licenses; see [Licenses](#licenses).

### 1. Recompute the paper tables from the released records (no API calls, no GPU)

```bash
uv run --frozen metc-artifact judge-tables --data-root metc-data --output-dir tables/judges
uv run --frozen metc-artifact human-tables --data-root metc-data --output-dir tables/human
uv run --frozen metc-artifact reference-matching --data-root metc-data \
  --output-dir tables/reference-matching --annotation-source authoritative_frozen
uv run --frozen metc-artifact reference-matching --data-root metc-data \
  --output-dir tables/reference-matching-exact --annotation-source authoritative_frozen \
  --span-match exact
```

These commands recompute the reference-matching, LLM-judge, and
human-evaluation tables from the frozen per-edit records, including bootstrap
confidence intervals. They do not re-run any system, teacher, or judge. On a
4-core CPU they take about 6 minutes in total.

### 2. Re-run the annotation systems (deterministic, no API calls)

```bash
uv run --frozen --extra paper python reproduce.py --accept-ltp-terms --no-llm
```

With `metc-data.zip` or `metc-data/` in the repository root, this:

1. validates the complete data artifact by file hash and content;
2. reconstructs the seven public benchmark settings from their upstream
   sources: CoNLL-2014 (annotators 0 and 1), MERLIN German, MERLIN Czech,
   KoLLA, UA-GEC, and MuCGEC (with the ChERRANT-based reference);
3. downloads and verifies the checkpoint (unless `--checkpoint-archive` or
   `--checkpoint-dir` is given), then runs METC, ERRANT, gERRANT, and
   JP-ERRANT from scratch;
4. compares the fresh annotations with the released ones, semantically, for
   sources, corrections, spans, replacements, and native and mapped labels;
5. recomputes reference matching from the fresh annotations; and
6. recomputes the English and German human-evaluation statistics.

It does not run the METC teacher or the LLM judges, so it does not regenerate
the judge comparison. That stage is reported as a known difference, not a PASS.

This command needs the same Linux/NVIDIA setup as the `paper` inference
profile, plus `git` and network access for the hash-pinned public datasets,
runtimes, and models. Without an NVIDIA GPU, use the portable profile:

```bash
uv run --frozen python reproduce.py \
  --accept-ltp-terms --systems all \
  --metc-inference-profile portable --no-llm
```

METC then runs in fp32 on the available device. Any difference from the
evaluated CUDA/bfloat16 annotations is reported with exact counts as
`KNOWN_DIFFERENCE`. It is never hidden or reported as a pass.

To run only some systems, use `--systems` with any of `metc`, `metc_teacher`,
`errant`, `gerrant`, and `jp_errant`. Selecting `metc_teacher` also selects
METC, because teacher labels reuse fresh METC spans. ChERRANT always runs,
because it builds the Chinese reference.

```bash
uv run --frozen python reproduce.py \
  --accept-ltp-terms --systems errant gerrant jp_errant --no-llm
```

ChERRANT uses the pinned `LTP/small` model. HIT-SCIR publishes nonstandard terms
for LTP, including restrictions on organizational and commercial use.
`--accept-ltp-terms` confirms that you have read those terms and that your use
is permitted. This repository does not redistribute the model or grant any
rights to it.

### 3. Full run, including the teacher and LLM judges (paid APIs)

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
uv run --frozen --extra paper python reproduce.py --accept-ltp-terms --llm
```

This adds the OpenAI teacher, the primary OpenAI judge, and the second Google
judge. It compares their outputs with the frozen runs and builds fresh judge
tables. The run is large: about 11,000 teacher jobs and up to 296,000 judge
jobs. Check your account limits and cost controls before starting.

Every request, response, retry, and comparison is recorded under
`metc-e2e-work/`, which must be empty for a new run. To resume an interrupted
run, repeat the same options with `--resume`:

```bash
uv run --frozen --extra paper python reproduce.py --accept-ltp-terms --llm \
  --resume --work-root metc-e2e-work --llm-run-id e2e
```

Resume first re-verifies the data, checkpoint, runtimes, selected systems, and
run settings. Completed teacher and judge requests are reused only if their
request fingerprints and stable edit IDs still match.

LLM outputs are not guaranteed to be identical across reruns, so a fresh run
measures agreement with the frozen run. It is not a bit-exact reproduction.
Human annotations are validated and recomputed, not recollected.

Run `uv run --frozen python reproduce.py --help` for options to supply local
data or checkpoint archives, pre-hydrated resources, a separate cache, or an
offline cache.

## Training

The training corpus is proprietary and is not distributed.
`training/recipe.json` records the training configuration, and
`training/training_corpus_summary.csv` records the aggregate label distribution.
The teacher annotation prompt and taxonomy in `prompts/` can be used to build
comparable METC-labeled data from other GEC corpora. See `MODEL_CARD.md`.
For the public release, taxonomy examples drawn from non-public text were
replaced with synthetic ones.

## Tests

```bash
uv run --frozen --group dev pytest
```

The checkpoint and data integration tests are skipped unless you point them at
the extracted assets:

```bash
METC_MODEL_PATH=metc-checkpoint METC_DATA_PATH=metc-data \
  uv run --frozen --group dev pytest
```

## Contributing

Contributions are welcome: bug reports, questions, documentation fixes, and
improvements to inference and tooling.

1. Open an issue to report a bug or discuss a larger change before you start.
2. Fork the repository and create a branch from `main`.
3. Run the linter and the tests before you open a pull request:

   ```bash
   uv run --frozen --group dev ruff check .
   uv run --frozen --group dev pytest
   ```

4. Open a pull request against `main` and describe what changed and why.

To keep the paper reproducible, the frozen evaluation records, prompts,
taxonomy, label mappings, and the pinned checkpoint and data hashes do not
change in place. A change to any of them is released as a new version.

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0, as described in Section 5 of `LICENSE`.

## Authors

- Gabriele Matiddi
- Tetiana Drobot
- Kostiantyn Omelianchuk

## Licenses

- **Code and model** (this repository and `metc-checkpoint.zip`): Apache
  License 2.0, Copyright 2026 Superhuman Platform Inc. See `LICENSE` and
  `NOTICE`. The base encoder, `jhu-clsp/mmBERT-small`, is declared MIT upstream
  (`model/MMBERT_NOTICE.md`).
- **Third-party components** bundled here keep their own terms. See
  `THIRD_PARTY_NOTICES.md`, `UNICODE_LICENSE.txt`, and
  `baselines/gerrant_de/LICENSE-MIT`.
- **Evaluation data** (`metc-data.zip`) is a **mixed-license** archive, Copyright
  2026 Superhuman Platform Inc. for the author-created parts:
  - The author-created system annotations, judge-output normalizations,
    manifests, and metadata are released under the Apache License 2.0.
  - Third-party benchmark text, gold annotations, and the separately licensed
    gERRANT dictionary keep their original licenses or terms of use (including
    CC BY-SA 4.0 and GPL). They are not relicensed under Apache-2.0.
  - The de-identified human-annotation records are supplied for research
    verification.

  The archive's `LICENSE` sets out these terms, and `DATASET_LICENSES.md` maps
  every released path to its source and terms.

## Citation

<!-- TODO(camera-ready): replace with the ACL Anthology entry once published. -->

```bibtex
@inproceedings{metc-2027,
  title     = {{METC}: Multilingual Error Type Classifier for Grammatical Error Correction},
  author    = {Matiddi, Gabriele and Drobot, Tetiana and Omelianchuk, Kostiantyn},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  url       = {TODO}
}
```
