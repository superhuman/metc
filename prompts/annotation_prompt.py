"""Prompt builder for the METC annotation teacher."""

from pathlib import Path


TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy_v2.md"
ANNOTATION_TEACHER_MODEL = "GPT-5.4"
ANNOTATION_SYSTEM_MESSAGE = "Return only valid JSON per the output contract. No extra text."
ANNOTATION_REQUEST_SETTINGS = {"reasoning_effort": "low"}

_ANNOTATION_TEMPLATE = """\
You are an expert error type classifier. Classify each numbered edit using the taxonomy.

## Taxonomy
{taxonomy}

## Output
Exactly one JSON line: {{"labels": [{labels_ph}], "sublabels": [{sublabels_ph}]}}
- Labels: Grammar / Punctuation / Spelling / Formatting / SentenceStructure / Other / Rephrasing
- Each sublabel must be valid under its label. For Other: Compound_Trivial / Compound_Moderate / Compound_Complex.
- Exact taxonomy names, case-sensitive. No extra text.

## Sentence pair
- Original: {src}
- Corrected: {tgt}

## Sentence with numbered edits
{numbered}

## Edit list
{edit_list}\
"""


def load_annotation_taxonomy() -> str:
    """Load the released taxonomy."""

    return TAXONOMY_PATH.read_text(encoding="utf-8")


def build_annotation_prompt(
    src: str,
    tgt: str,
    numbered: str,
    edits: list[dict],
    taxonomy: str | None = None,
) -> str:
    if taxonomy is None:
        taxonomy = load_annotation_taxonomy()
    n = len(edits)
    labels_ph = ", ".join(f'"<LABEL_{i}>"' for i in range(1, n + 1))
    sublabels_ph = ", ".join(f'"<SUBLABEL_{i}>"' for i in range(1, n + 1))
    edit_list = "\n".join(
        f"{i}. {edit['source'] or '(empty)'} => {edit['target'] or '(empty)'}"
        for i, edit in enumerate(edits, 1)
    )
    return _ANNOTATION_TEMPLATE.format(
        taxonomy=taxonomy,
        labels_ph=labels_ph,
        sublabels_ph=sublabels_ph,
        src=src,
        tgt=tgt,
        numbered=numbered,
        edit_list=edit_list,
    )


def build_annotation_messages(
    src: str,
    tgt: str,
    numbered: str,
    edits: list[dict],
    taxonomy: str | None = None,
) -> list[dict[str, str]]:
    """Build the exact system and user messages used by the annotation teacher."""

    return [
        {"role": "system", "content": ANNOTATION_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_annotation_prompt(src, tgt, numbered, edits, taxonomy),
        },
    ]
