"""
gerrant_de — port of Boyd 2018 German ERRANT (aclweb.org/anthology/W18-6111)
to modern spaCy 3.x.

Faithful reimplementation:
- Edit alignment: unchanged (Damerau-Levenshtein with linguistic cost via align_text.py).
- Edit classification: Boyd's cat_rules_de.py rules, unchanged logic. Only spaCy
  API calls updated (integer POS enum → string; nlp.tagger direct call → pipeline iter).
- Lemma: now from spaCy's de_core_news_sm built-in lemmatizer (TreeTagger removed).

USAGE:
    .venv-gerrant/bin/python gerrant_de.py \\
        -orig orig.txt -cor cor.txt -out out.m2

Input files: one sentence per line, pre-tokenized (space-separated).
"""

import argparse
import os
import sys

# Ensure local imports resolve when run directly
_here = os.path.dirname(os.path.realpath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import spacy
from nltk.stem.snowball import GermanStemmer

import align_text
import toolbox
import cat_rules_de


def main(args):
    basename = os.path.dirname(os.path.realpath(__file__))
    print("Loading resources...", file=sys.stderr)

    # Port: spacy.load("de") (spaCy 1.x alias) → explicit model name (spaCy 3.x)
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    stemmer = GermanStemmer()
    word_list = toolbox.loadDictionary(os.path.join(basename, "resources", "de_DE-large.txt"))
    tag_map = toolbox.loadTagMap(os.path.join(basename, "resources", "de-stts_map"), lang="de")

    with open(args.out, "w", encoding="utf-8", newline="\n") as out_m2, \
         open(args.orig, encoding="utf-8") as orig_f, \
         open(args.cor, encoding="utf-8") as cor_f:
        for orig_sent, cor_sent in zip(orig_f, cor_f, strict=True):
            orig_sent = orig_sent.strip()
            cor_sent = cor_sent.strip()
            # Note: -tok means "run spaCy tokenizer" (input NOT pre-tokenized), matching Boyd.
            pre_tokenized = not args.tok
            proc_orig = toolbox.applySpacy(orig_sent, nlp, tokenized=pre_tokenized)
            proc_cor = toolbox.applySpacy(cor_sent, nlp, tokenized=pre_tokenized)

            out_m2.write("S " + " ".join(tok.text for tok in proc_orig) + "\n")

            if orig_sent == cor_sent:
                out_m2.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            else:
                if args.ann:
                    out_m2.write("O " + toolbox.formatAnnotation(proc_orig) + "\n")
                    out_m2.write("C " + toolbox.formatAnnotation(proc_cor) + "\n")
                auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, nlp, args)
                for auto_edit in auto_edits:
                    cat = cat_rules_de.autoTypeEdit(auto_edit, proc_orig, proc_cor,
                                                    word_list, tag_map, nlp, stemmer)
                    auto_edit[2] = cat
                    out_m2.write(toolbox.formatEdit(auto_edit) + "\n")
            out_m2.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Port of Boyd 2018 gERRANT to spaCy 3.x. German parallel text → M2.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-orig", required=True, help="Original (learner) text file, one sentence per line.")
    parser.add_argument("-cor", required=True, help="Corrected text file, one sentence per line.")
    parser.add_argument("-out", required=True, help="Output M2 path.")
    parser.add_argument("-lev", action="store_true", help="Plain Levenshtein alignment.")
    parser.add_argument("-merge", choices=["rules", "all-split", "all-merge", "all-equal"], default="rules")
    parser.add_argument("-tok", action="store_true", help="Input is NOT pre-tokenized; run spaCy tokenizer.")
    parser.add_argument("-ann", action="store_true", help="Emit 'O' / 'C' annotation lines.")
    parser.add_argument("-lang", default="de", help=argparse.SUPPRESS)  # fixed
    main(parser.parse_args())
