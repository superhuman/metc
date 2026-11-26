# Third-party notices

Third-party terms apply independently of the Apache License 2.0 terms for
METC material created by the authors.

## Released or redistributed components

| Component | Use and distribution status | License |
|---|---|---|
| `jhu-clsp/mmBERT-small` encoder | Base encoder for the externally hosted fine-tuned checkpoint | Upstream repository declares MIT |
| German ERRANT extensions | Modified Python baseline code in `baselines/gerrant_de/` | MIT; full upstream notice in `baselines/gerrant_de/LICENSE-MIT` |
| MuCGEC ChERRANT tokenizer logic | `src/metc_artifact/datasets/mucgec.py` contains a modified, embedded form of the pinned ChERRANT tokenizer | Apache-2.0 |
| diff-match-patch word-diff adaptation | `src/metc/extractor/metc_diff_extension.py` adapts diff-match-patch's line-to-character encoding to released token boundaries | Apache-2.0 |
| Unicode Emoji 16.0 `emoji-sequences.txt` | Bundled data for emoji-aware edit boundaries | Unicode License v3 (`Unicode-3.0`) |

The mmBERT source is pinned to commit
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`. Its repository declares MIT in
the model-card metadata but does not provide a standalone license file or full
copyright notice.

The German ERRANT port is pinned to `adrianeboyd/errant` commit
`92a86d0722536f4b2c6a4a10037b64f3955ad1b5`. Its source README contains the
MIT grant and copyright notice for Christopher Bryant and Mariano Felice.
`baselines/gerrant_de/MODIFICATIONS.md` identifies the spaCy 3 compatibility
changes. `rdlextra.py` also retains its embedded notice for the underlying
Kyle Gorman implementation.

The embedded ChERRANT tokenizer logic is derived from MuCGEC commit
`39221934ee4ccf07801879136a75dd7e11319977`. The released version changes the
model load to the pinned `LTP/small` snapshot, uses the LTP 4.2 pipeline API,
and defers the LTP import until word tokenization is requested. MuCGEC and its
bundled ChERRANT code are licensed under Apache License 2.0.

`metc_diff_extension.py` retains the structure of diff-match-patch's
line-to-character encoding while replacing lines with METC token boundaries.
The underlying diff-match-patch implementation and this adapted logic are
licensed under Apache License 2.0.

The bundled `emoji-sequences.txt` retains its Unicode copyright and
terms-of-use header. The complete Unicode License v3 copyright and permission
notice is included as `UNICODE_LICENSE.txt`.

## Runtime dependencies not bundled in the software archive

| Component | License |
|---|---|
| PyTorch | BSD-3-Clause |
| Hugging Face Hub | Apache-2.0 |
| Hugging Face Transformers | Apache-2.0 |
| Hugging Face Tokenizers | Apache-2.0 |
| Hugging Face Safetensors | Apache-2.0 |
| diff-match-patch | Apache-2.0 |
| spaCy | MIT |
| `de_core_news_sm` 3.8.0 | MIT; model sources also carry their own notices |
| NLTK | Apache-2.0 |
| Python `regex` | Apache-2.0 AND CNRI-Python |

## Runtime-downloaded model assets

The Code artifact does not redistribute these model files. Hydration downloads
and verifies only the pinned assets needed by the evaluation.

- The Chinese JP-ERRANT path downloads
  `UD_Chinese-GSDLTP.tar.gz` from Zenodo record
  [15096936](https://zenodo.org/records/15096936), created by Yige Chen and
  Jungyeul Park. The record declares CC BY 4.0. Hydration verifies archive
  SHA-256
  `bcf6673629dd8574402d1eb9124a1e19a12f75d299abe4617d84e757404d91ed`
  and extracts only the pinned tokenizer and POS model members.
- ChERRANT downloads `LTP/small` revision
  `dea87bf0da3c5d054bbd426a78d7fa9efd347799` from Hugging Face. That model
  repository does not declare a standard SPDX license. Its model card carries
  the HIT-SCIR LTP terms: academic and individual research use is free,
  commercial use and use by other organizations require payment, and published
  work should acknowledge LTP. Users must review those upstream terms before
  hydration. The reproduction command therefore requires the explicit
  `--accept-ltp-terms` confirmation and does not download the model otherwise.
- JP-ERRANT uses Stanza 1.10.1 and the exact model inventory pinned in
  `runtime-locks/stanza-models.json`. Stanza code is Apache-2.0, but Stanford
  states that licensing for models built from UD treebanks is unclear and
  offers only Stanford's own rights in the language packs under the Open Data
  Commons Attribution License 1.0. The model packs are acquired directly from
  Stanza and are not redistributed in this artifact. The pinned packages are
  English `combined`, German `combined`, Czech `pdt`, Korean `kaist`,
  Ukrainian `iu`, and Chinese `gsdsimp`.

## User-acquired tokenizer source

The checkpoint archive omits `tokenizer.json`, `tokenizer_config.json`, and
`special_tokens_map.json`. `metc prepare-tokenizer` obtains four exact source
files directly from `jhu-clsp/mmBERT-small` revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`, verifies their hashes, adds the two
METC marker tokens, and verifies the reconstructed files against the evaluated
checkpoint. The upstream model card identifies the tokenizer as Gemma 2.
`TOKENIZER_MODIFICATIONS.md` records the complete provenance and transformation.

## Deliberately excluded resources

The following inputs are not redistributed:

- TreeTagger and its German model. The spaCy 3 port does not use them.
- Stanza model packs and `jp_errant_bea` code or models.
- Any Gemma model weights or tokenizer files. Users obtain the four pinned
  tokenizer source files directly from the mmBERT repository and reconstruct
  the three runtime tokenizer files locally as documented above.

The Apache-licensed Code artifact does not contain `de_DE-large.txt`. The
mixed-license Data artifact distributes that igerman98-derived dictionary,
the pinned source archive, the upstream copyright notice, and the complete
GPLv2 and GPLv3 texts as a separate
`GPL-2.0-only OR GPL-3.0-only` runtime component.

## Upstream pages

- mmBERT: <https://huggingface.co/jhu-clsp/mmBERT-small/tree/abc32620dd4f6ab06f5fbe905dc25f310618e09f>
- MuCGEC/ChERRANT: <https://github.com/HillZhang1999/MuCGEC/tree/39221934ee4ccf07801879136a75dd7e11319977/scorers/ChERRANT>
- GSDLTP model record: <https://zenodo.org/records/15096936>
- LTP/small pinned model: <https://huggingface.co/LTP/small/tree/dea87bf0da3c5d054bbd426a78d7fa9efd347799>
- LTP upstream terms: <https://github.com/HIT-SCIR/ltp>
- German ERRANT: <https://github.com/adrianeboyd/errant/tree/92a86d0722536f4b2c6a4a10037b64f3955ad1b5>
- Boyd (2018): <https://aclanthology.org/W18-6111/>
- igerman98: <https://www.j3e.de/ispell/igerman98/>
- STTS tag inventory: <https://www.ims.uni-stuttgart.de/documents/ressourcen/lexika/tagsets/stts-table.html>
- Universal POS tags: <https://universaldependencies.org/u/pos/>
- PyTorch: <https://github.com/pytorch/pytorch/blob/main/LICENSE>
- Hugging Face Hub: <https://github.com/huggingface/huggingface_hub/blob/main/LICENSE>
- Transformers: <https://github.com/huggingface/transformers/blob/main/LICENSE>
- Tokenizers: <https://github.com/huggingface/tokenizers/blob/main/LICENSE>
- Safetensors: <https://github.com/huggingface/safetensors/blob/main/LICENSE>
- diff-match-patch: <https://github.com/google/diff-match-patch/blob/master/LICENSE>
- spaCy: <https://github.com/explosion/spaCy/blob/master/LICENSE>
- NLTK: <https://github.com/nltk/nltk/blob/develop/LICENSE.txt>
- regex: <https://github.com/mrabarnett/mrab-regex>
- Stanza model terms: <https://stanfordnlp.github.io/stanza/performance.html>
- Open Data Commons Attribution 1.0: <https://opendatacommons.org/licenses/by/1-0/>
- Unicode terms: <https://www.unicode.org/terms_of_use.html>
