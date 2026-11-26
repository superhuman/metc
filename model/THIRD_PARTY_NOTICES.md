# Checkpoint third-party notices

The METC checkpoint is fine-tuned from `jhu-clsp/mmBERT-small` revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`. The upstream repository declares
MIT in its model-card metadata but does not provide a standalone license file
or full copyright notice.

Tokenizer files are not included. `metc prepare-tokenizer` obtains the pinned
source files directly from the upstream repository and adds only `[EDIT]` and
`[/EDIT]`. The upstream model card identifies the tokenizer as Gemma 2.

Apache License 2.0 applies to the author-created checkpoint material as stated
in `LICENSE`; it does not replace terms that apply to upstream components.
`MMBERT_NOTICE.md` records the pinned upstream source, MIT declaration, and
requested citation.
