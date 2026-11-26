# METC checkpoint

This archive contains the four model files for the 42-label METC classifier:
`config.json`, two safetensor shards, and their index.

Tokenizer files are intentionally omitted. With the separate METC software
artifact installed, prepare the exact tokenizer once:

```bash
metc prepare-tokenizer --model /path/to/metc-checkpoint
```

The command downloads a pinned `jhu-clsp/mmBERT-small` tokenizer snapshot,
verifies its hashes, adds the two METC edit markers, and verifies the generated
files. Inference is offline after preparation. Usage, model details, taxonomy,
prompts, and evaluation code are in the software artifact.

The author-created checkpoint material is distributed under Apache License
2.0. The upstream mmBERT declaration remains MIT. See `LICENSE`,
`MMBERT_NOTICE.md`, and `THIRD_PARTY_NOTICES.md`.
