# German ERRANT baseline reproduction

This optional component regenerates the pinned German gERRANT M2 output. It is
not imported by standalone METC inference. The full reproduction command hydrates
its pinned runtime and resources, executes it on MERLIN, and compares the
resulting annotations with the complete Data artifact.

The generated M2 contains 2,880 non-no-op edits. The historical evaluation
materialization contains 2,879: its source-only join encountered one duplicated
source string and selected the first, zero-edit occurrence, excluding the
punctuation edit attached to the second occurrence. The frozen evaluation data
preserves that historical behavior rather than silently changing a paper
result. `manifest.json` records both counts and the one-edit exclusion.

## Provenance and license boundary

The port is based on the German ERRANT extensions released with Boyd (2018),
pinned to `adrianeboyd/errant` commit
`92a86d0722536f4b2c6a4a10037b64f3955ad1b5`. That repository's README contains
the MIT grant retained verbatim in `LICENSE-MIT`. `MODIFICATIONS.md` records
the spaCy 3 compatibility changes in this port.

The dictionary runtime resource is kept outside the Apache-licensed Code
artifact:

- `de_DE-large.txt` is derived from igerman98, whose dictionaries and word
  lists are GPL version 2 or 3. The mixed-license Data artifact distributes the
  exact derived dictionary, pinned source archive, copyright notice, and both
  GPL texts as a separate component. `prepare_resources.py` accepts that
  dictionary only after exact size and SHA-256 verification. It can also derive
  the same file from the archive when `make` and Hunspell's `unmunch` are
  installed.
- `de-stts_map` is generated from an author-created functional compatibility
  table embedded in `prepare_resources.py`. It maps the published STTS tag
  inventory to universal POS categories and includes the two gERRANT
  compatibility aliases `NNE -> PNOUN` and `PROAV -> ADV`. The generated
  56-row file is required to match the evaluated SHA-256 exactly. No
  TuebaUdConverter source or derived file is downloaded or distributed.

## Exact setup

Use CPython 3.11.13. The direct runtime packages and German spaCy model are
pinned in `requirements.txt`:

```bash
python3.11 -m venv .venv-gerrant
.venv-gerrant/bin/python -m pip install -r requirements.txt
```

With the extracted Data artifact, prepare both resources without host build
tools:

```bash
.venv-gerrant/bin/python prepare_resources.py --output-dir resources \
  --dictionary /path/to/metc-data/resources/gerrant/de_DE-large.txt
```

To derive the dictionary independently instead, install `make` and Hunspell's
`unmunch`, then replace `--dictionary` with
`--igerman-archive /path/to/igerman98-20161207.tar.bz2` or omit both options to
download the pinned archive.

Run the baseline on one pre-tokenized source and correction per line:

```bash
.venv-gerrant/bin/python gerrant_de.py \
  -orig merlin.orig.txt -cor merlin.th1.txt -out gerrant_port_de.m2
```

`manifest.json` pins every code and resource digest, the tested runtime, the
expected full-benchmark M2 digest and counts, and the historical one-edit
materialization exclusion. A mismatched generated resource or M2 file is not
the reviewed baseline output.

## Citation

Adriane Boyd. 2018. Using Wikipedia Edits in Low Resource Grammatical Error
Correction. Proceedings of W-NUT 2018. <https://aclanthology.org/W18-6111/>
