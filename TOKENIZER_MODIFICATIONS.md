# Tokenizer provenance and modifications

The hosted checkpoint archive does not distribute tokenizer files. Users
reconstruct the exact evaluated tokenizer with:

```bash
metc prepare-tokenizer --model /path/to/metc-checkpoint
```

The command downloads exactly `config.json`, `special_tokens_map.json`,
`tokenizer.json`, and `tokenizer_config.json` from
`jhu-clsp/mmBERT-small` revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`. It verifies the pinned byte size
and SHA-256 of all four source files before loading them. The upstream
repository identifies this tokenizer as Gemma 2.

The pinned upstream snapshot already contains the mask-token behavior used by
the evaluated checkpoint. METC adds only `[EDIT]` and `[/EDIT]` as special
tokens with IDs 256000 and 256001, then performs the same two-pass serialization
used for the evaluated export. The preparation tool requires Transformers
4.57.3 and Tokenizers 0.22.0, then verifies the exact byte size and SHA-256 of
`tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` before
installing them beside the model weights.

Apart from the two METC marker tokens, the checkpoint vocabulary and all
580,604 merges match the pinned mmBERT tokenizer after serialization.
`model/checkpoint.json` records the source provenance and hashes and the exact
hashes of all three generated files. Inference is offline after preparation.

These modifications do not replace or supersede terms that apply to the
upstream tokenizer. Users obtain the four source files directly from the
pinned upstream repository.
