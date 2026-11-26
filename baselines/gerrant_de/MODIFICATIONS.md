# German ERRANT port modifications

The Python files in this directory derive from the MIT-licensed German ERRANT
implementation in `adrianeboyd/errant` at commit
`92a86d0722536f4b2c6a4a10037b64f3955ad1b5`. The complete upstream MIT notice
is retained in `LICENSE-MIT`.

The release port makes the following compatibility changes:

- `align_text.py` replaces spaCy 1.x integer POS constants and the removed
  morphology lemmatizer API with spaCy 3.x string POS values and pipeline
  lemmas.
- `cat_rules_de.py` removes an unused spaCy 1.x import. The German
  classification rules are otherwise unchanged.
- `toolbox.py` runs a pre-built `Doc` through the spaCy 3.x pipeline, retains
  the original punctuation-tag override, and removes the TreeTagger path.
- `gerrant_de.py` is a Python 3/spaCy 3 command-line wrapper for the German
  alignment and classification path.
- `rdlextra.py` retains the upstream alignment implementation and its embedded
  MIT notice; the release copy only normalizes the file ending.
- `cat_rules_de.py` is stored with LF line endings instead of the upstream
  checkout's CRLF line endings.

No German dictionary, STTS mapping, TreeTagger model, spaCy model pack, Gemma
tokenizer package, or `jp_errant_bea` code is included in this component.
