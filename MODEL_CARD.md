# METC model card

## Model summary

METC is a multilingual grammatical-error-type classifier. It receives a source
sentence and its correction, extracts one or more edits, and assigns each edit
one of 42 shared labels.

- Base encoder: `jhu-clsp/mmBERT-small`
- Architecture: `ModernBertForTokenClassification`
- Parameters: 140,658,474
- Output labels: 42
- Edit markers: `[EDIT]` and `[/EDIT]`
- Fine-tuning input limit: 256 tokens
- Reference inference limit: 512 tokens
- Model capacity: 8,192 positions

The fine-tuning limit and reference inference limit are intentionally reported
separately. Training examples were truncated to 256 tokens. The released
inference code uses the evaluated 512-token serving limit. The endpoint first
bounds each input to 3,584 Unicode characters, retains predictions for
`[EDIT]` markers that survive right-side tokenization, and assigns
`Other/Compound_Complex` to any remaining trailing edits.

## Training data and supervision

The student was trained on 1,529,258 examples and selected using 169,918
examples from an internal held-out validation split. The source corpus contains
approximately 1.7 million source-correction pairs across more than 200
languages. A separate annotation teacher assigned the 42 METC labels to edits
extracted from those pairs.

The three reported source tiers contain approximately 438,000 English rows,
450,000 rows from the five higher-resource non-English languages, and 811,000
long-tail rows. Exact split counts and the rounded tier counts are recorded in
`training/recipe.json`.

The source corpus is proprietary and is not distributed. The annotation prompt
and taxonomy are included so METC-labeled examples can be constructed from
other multilingual grammatical-error-correction corpora. The prompts used by
upstream correction systems to create target rewrites are not included.

## Fine-tuning and checkpoint selection

- Optimizer: AdamW
- Learning rate: 2e-5
- Per-device batch size: 16 for training, 64 for evaluation
- Warmup ratio: 0.1
- Weight decay: 0.01
- Precision: fp16
- Random seed: 42
- Selection metric: macro-F1 on the internal held-out split
- Selected checkpoint: step 23,000, macro-F1 0.7583
- Training stop: step 26,000 after three evaluations without improvement

The paper evaluation benchmarks were not used for checkpoint selection.

`training/recipe.json` records the released token-classification configuration,
including the nine source roles, global split procedure, and fine-tuning
parameters. The proprietary examples themselves are not distributed, so this
records the run configuration but does not make the original training corpus
reproducible.

## Intended use

METC is intended for multilingual error analysis, research, and assistive
annotation. It can compare or normalize edit types across languages without a
language-specific linguistic pipeline.

## Limitations

- Exact reproduction of training is unavailable because the source corpus is
  not distributed.
- A shared taxonomy may omit distinctions important to a particular language,
  domain, or annotation scheme.
- Labels can be wrong even when the underlying correction is valid.
- The released evaluation covers fewer languages than the training corpus.
- The model has not been validated for pedagogical assessment or other
  high-stakes decisions.

Applications that expose labels to users should validate the relevant language
and domain and retain human review where incorrect labels could cause harm.

## Reproducibility

`model/checkpoint.json` records the exact hosted archive size and hash as well
as the checkpoint filenames, byte sizes, and SHA-256 digests. The hosted
archive contains the configuration, two safetensor shards, and their index; it
does not redistribute `tokenizer.json`, `tokenizer_config.json`, or
`special_tokens_map.json`.

Run `metc prepare-tokenizer --model /path/to/metc-checkpoint` once after
extracting the archive. It downloads and hash-verifies four source files from
`jhu-clsp/mmBERT-small` revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`, adds `[EDIT]` and `[/EDIT]` as
IDs 256000 and 256001, serializes twice with Transformers 4.57.3 and Tokenizers
0.22.0, and verifies the final three files byte-for-byte against the evaluated
checkpoint. The package then validates all seven runtime files, the parameter
count, architecture, label inventory, marker-token IDs, and maximum-length
behavior before offline inference.

The evaluated serving path used Python 3.12, PyTorch 2.9.0, Transformers
4.57.3, Accelerate 1.12.0, FlashAttention 2.8.2, CUDA 12.8.1, bfloat16, and
fixed 512-token right padding. The `paper` inference profile requires those
CUDA semantics and fails rather than silently changing hardware profile. The
default `portable` fp32 profile can change a small number of hardware-sensitive
near-tie predictions;
it is not presented as bit-exact reproduction of the serving run. The complete
Data artifact contains the evaluated benchmark text, references, system and
human annotations, and frozen teacher and judge outputs. The Code artifact
independently reconstructs the public data, executes every evaluation system,
compares fresh annotations with those authoritative records, and reruns the
teacher and judges when API keys are supplied. See `README.md` for the
full execution command and its hardware requirements.

The German gERRANT baseline is separate from METC inference. Its released port
records the exact spaCy 3 compatibility changes and runtime pins. The
GPL-derived dictionary is distributed only in the separately mixed-license
Data artifact, together with its source archive and notices. It is not part of
the Apache-licensed Code artifact. The STTS mapping, whose upstream source has
no visible license, is not redistributed; its source and derived hashes are
recorded for local reconstruction in `baselines/gerrant_de/`.

## License

The author-created model and code are released under Apache License 2.0. See
`LICENSE` for the complete terms and `THIRD_PARTY_NOTICES.md` for the separate
terms covering the base model, libraries, and Unicode data used by the package.
