"""
Port of Boyd 2018 errant/scripts/toolbox.py to spaCy 3.x.

Changes vs original:
- applySpacy rewritten for spaCy 3.x pipeline API (no more nlp.tagger/nlp.parser direct call).
- TreeTagger path removed entirely — modern de_core_news_sm provides lemmas in-pipeline.
- loadTagMap: kept intact (reads STTS->UPOS map file).
"""

from spacy.tokens import Doc
import regex as re


def loadDictionary(path):
	return set(open(path).read().split())


def loadTagMap(path, lang="de"):
	"""Port: second arg was argparse Namespace in original; collapsed to a lang string."""
	map_dict = {}
	open_file = open(path).readlines()
	for line in open_file:
		line = line.strip().split("\t")
		map_dict[line[0]] = line[1].strip()
		if lang == 'en':
			if line[1].strip() == "ADP":
				map_dict[line[0]] = "PREP"
			elif line[1].strip() == "PROPN":
				map_dict[line[0]] = "NOUN"
	if lang == 'en':
		map_dict['""'] = "PUNCT"
		map_dict["SP"] = "SPACE"
		map_dict["ADD"] = "X"
		map_dict["GW"] = "X"
		map_dict["NFP"] = "X"
		map_dict["XX"] = "X"
	return map_dict


def applySpacy(input_sent, nlp, tokenized=True):
	"""
	Port: spaCy 1.x API `nlp.tagger(sent)` / `nlp.parser(sent)` no longer works.
	spaCy 3.x: iterate through `nlp.pipeline` components on a pre-built Doc.
	TreeTagger path removed — lemmas now come from de_core_news_sm's built-in lemmatizer.
	"""
	if not tokenized:
		return nlp(input_sent)

	words = input_sent.split()
	if not words:
		return Doc(nlp.vocab, words=[], spaces=[])
	# Port fix: match Boyd's default Doc(vocab, tokens) behaviour (trailing space on every token).
	spaces = [True] * len(words)
	doc = Doc(nlp.vocab, words=words, spaces=spaces)
	for _, pipe in nlp.pipeline:
		doc = pipe(doc)

	# Port fix: preserve Boyd's punct-override — force $( for inline punctuation tokens,
	# whether or not the spaCy pipeline already assigned a tag. Mirrors Boyd toolbox.py:120-124
	# which ran after treetagger assignment.
	for tok in doc:
		if tok.tag_ == "":
			tok.tag_ = "$(" if re.match(r'^\p{P}+$', tok.text) else "XX"
		elif re.match(r'^\p{P}+$', tok.text):
			tok.tag_ = "$("
	return doc


def formatEdit(edit, coder_id=0):
	span = " ".join(["A", str(edit[0]), str(edit[1])])
	return "|||".join([span, edit[2], edit[3], "REQUIRED", "-NONE-", str(coder_id)])


def formatAnnotation(sent):
	ann = ["/".join([w.orth_, w.tag_, w.pos_, w.lemma_, w.dep_]) for w in sent]
	return " ".join(ann)
