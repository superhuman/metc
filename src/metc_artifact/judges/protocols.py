"""Public rerun protocols for METC's two judges and annotation teacher."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from metc.mapping import load_label_mapping


QUALITY_LEVELS = frozenset({"Good", "Acceptable", "Bad"})

ACCEPTABILITY_SYSTEM_MESSAGE = (
    "Return only valid JSON per the contract; no markdown, no extra text."
)
ACCEPTABILITY_PROMPT = """You are a senior linguist evaluating one error-correction edit.

Language: {language}
Source sentence: "{source}"
Corrected sentence: "{target}"
The edit being judged: {edit_diff}
The classifier described this edit as: {label_gloss}

Rate the classifier on two independent axes. For each, choose ONE of:
  Good        — fully appropriate
  Acceptable  — defensible but not the best choice (span is not semantically clean, or a coarser super-type description)
  Bad         — wrong or misleading

span_quality:
  Good       = the span exactly covers the error, no more no less
  Acceptable = the span covers the error but is semantically confusing or not precise 
  Bad        = the span includes unrelated text that makes it unclear what error the edit is fixing

label_quality:
  Good       = the description is the most precise correct interpretation of this edit
  Acceptable = the description is a coarser but still valid interpretation
  Bad        = the description is wrong, irrelevant, or contradicts what the edit does

Return only a JSON object, no prose, no markdown:
{{
  "span_quality": "Good" | "Acceptable" | "Bad",
  "label_quality": "Good" | "Acceptable" | "Bad",
  "span_reason": "<one short sentence citing the source span>",
  "label_reason": "<one short sentence citing why the description fits or does not>"
}}
"""

TEACHER_SYSTEM_MESSAGE = "Return only valid JSON per the output contract. No extra text."
TEACHER_PROMPT = """\
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


class ProtocolError(ValueError):
    """Raised when a job or provider response violates the released protocol."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_acceptability_prompt(job: Mapping[str, Any]) -> str:
    """Render the exact paper prompt from one stable edit job."""

    fields = {
        "language": job.get("lang", job.get("language")),
        "source": job.get("src", job.get("source")),
        "target": job.get("tgt", job.get("target")),
        "edit_diff": job.get("edit_diff"),
        "label_gloss": job.get("label_gloss"),
    }
    missing = [key for key, value in fields.items() if not isinstance(value, str)]
    if missing:
        raise ProtocolError("acceptability job lacks string fields: " + ", ".join(sorted(missing)))
    edit_id = job.get("edit_id")
    if not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:"):
        raise ProtocolError("acceptability job lacks a stable metc-edit-v1 edit_id")
    return ACCEPTABILITY_PROMPT.format(**fields)


def _extract_json_object(content: str) -> Mapping[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        pieces = stripped.split("```")
        stripped = pieces[1] if len(pieces) >= 3 else stripped.replace("```", "")
        if stripped.lower().lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ProtocolError("provider content is not a JSON object") from None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as error:
            raise ProtocolError("provider content is not a JSON object") from error
    if not isinstance(value, Mapping):
        raise ProtocolError("provider content is not a JSON object")
    return value


def parse_acceptability_content(content: str) -> dict[str, str]:
    value = _extract_json_object(content)
    span_quality = str(value.get("span_quality", "")).strip().capitalize()
    label_quality = str(value.get("label_quality", "")).strip().capitalize()
    if span_quality not in QUALITY_LEVELS or label_quality not in QUALITY_LEVELS:
        raise ProtocolError("acceptability response contains an invalid quality rating")
    span_reason = value.get("span_reason")
    label_reason = value.get("label_reason")
    if not isinstance(span_reason, str) or not isinstance(label_reason, str):
        raise ProtocolError("acceptability response lacks string rationales")
    return {
        "span_quality": span_quality,
        "label_quality": label_quality,
        "span_reason": span_reason,
        "label_reason": label_reason,
    }


def default_taxonomy_path() -> Path:
    candidates = []
    configured = os.environ.get("METC_TEACHER_TAXONOMY_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            Path(__file__).resolve().parents[3] / "prompts" / "taxonomy_v2.md",
            Path.cwd() / "prompts" / "taxonomy_v2.md",
            Path.cwd() / "software" / "prompts" / "taxonomy_v2.md",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ProtocolError(
        "cannot locate the released teacher taxonomy; pass taxonomy_path or set "
        "METC_TEACHER_TAXONOMY_PATH"
    )


def load_teacher_taxonomy(path: Path | None = None) -> str:
    taxonomy_path = path or default_taxonomy_path()
    try:
        taxonomy = taxonomy_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProtocolError(f"cannot read teacher taxonomy: {taxonomy_path}") from error
    validate_teacher_taxonomy(taxonomy)
    return taxonomy


@lru_cache(maxsize=4)
def validate_teacher_taxonomy(taxonomy: str) -> None:
    expected = {f"{label}/{sublabel}" for label, sublabel in load_label_mapping().values()}
    if taxonomy_pairs(taxonomy) != expected:
        raise ProtocolError("teacher taxonomy labels differ from the released label mapping")


def _teacher_edits(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_edits = job.get("edits")
    if not isinstance(raw_edits, Sequence) or isinstance(raw_edits, (str, bytes)):
        raise ProtocolError("teacher job edits must be a nonempty sequence")
    edits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw_edit in enumerate(raw_edits, 1):
        if not isinstance(raw_edit, Mapping):
            raise ProtocolError(f"teacher edit {position} is not an object")
        edit = dict(raw_edit)
        edit_id = edit.get("edit_id")
        if (
            not isinstance(edit_id, str)
            or not edit_id.startswith("metc-edit-v1:")
            or edit_id in seen
        ):
            raise ProtocolError(f"teacher edit {position} lacks a unique stable edit_id")
        if not isinstance(edit.get("source"), str) or not isinstance(edit.get("target"), str):
            raise ProtocolError(f"teacher edit {position} lacks source/target text")
        span = edit.get("span")
        coordinate_pairs: list[tuple[Any, Any]] = []
        if isinstance(span, Mapping):
            coordinate_pairs.append((span.get("start"), span.get("end")))
        coordinate_pairs.extend(
            (edit.get(start_key), edit.get(end_key))
            for start_key, end_key in (
                ("start", "end"),
                ("edit_start", "edit_end"),
                ("orig_start", "orig_end"),
                ("char_start", "char_end"),
            )
        )
        if not any(
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end >= start
            for start, end in coordinate_pairs
        ):
            raise ProtocolError(f"teacher edit {position} lacks a valid fixed span")
        seen.add(edit_id)
        edits.append(edit)
    if not edits:
        raise ProtocolError("teacher job must contain at least one fixed student edit")
    return edits


def render_teacher_prompt(
    job: Mapping[str, Any],
    *,
    taxonomy_path: Path | None = None,
    taxonomy: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render a teacher prompt with the current taxonomy and fixed student edits."""

    source = job.get("src", job.get("source"))
    target = job.get("tgt", job.get("target"))
    numbered = job.get("numbered", job.get("numbered_sentence"))
    if not all(isinstance(value, str) for value in (source, target, numbered)):
        raise ProtocolError("teacher job lacks source, target, or numbered sentence text")
    edits = _teacher_edits(job)
    if taxonomy is None:
        taxonomy = load_teacher_taxonomy(taxonomy_path)
    labels_ph = ", ".join(f'"<LABEL_{index}>"' for index in range(1, len(edits) + 1))
    sublabels_ph = ", ".join(f'"<SUBLABEL_{index}>"' for index in range(1, len(edits) + 1))
    edit_list = "\n".join(
        f"{index}. {edit['source'] or '(empty)'} => {edit['target'] or '(empty)'}"
        for index, edit in enumerate(edits, 1)
    )
    prompt = TEACHER_PROMPT.format(
        taxonomy=taxonomy,
        labels_ph=labels_ph,
        sublabels_ph=sublabels_ph,
        src=source,
        tgt=target,
        numbered=numbered,
        edit_list=edit_list,
    )
    return prompt, edits


@lru_cache(maxsize=4)
def taxonomy_pairs(taxonomy: str) -> frozenset[str]:
    current_label: str | None = None
    pairs: set[str] = set()
    for line in taxonomy.splitlines():
        top_level = re.match(r"^\* ([A-Za-z]+)(?::|$)", line)
        if top_level:
            current_label = top_level.group(1)
            continue
        sublabel = re.match(r"^  \* ([A-Za-z_]+)\s+—", line)
        if sublabel and current_label is not None:
            pairs.add(f"{current_label}/{sublabel.group(1)}")
    return frozenset(pairs)


def parse_teacher_content(
    content: str,
    edits: Sequence[Mapping[str, Any]],
    *,
    taxonomy: str,
) -> dict[str, Any]:
    value = _extract_json_object(content)
    labels = value.get("labels")
    sublabels = value.get("sublabels")
    if not isinstance(labels, list) or not isinstance(sublabels, list):
        raise ProtocolError("teacher response lacks label and sublabel lists")
    if len(labels) != len(edits) or len(sublabels) != len(edits):
        raise ProtocolError("teacher response length differs from the fixed student edits")
    allowed = taxonomy_pairs(taxonomy)
    outputs: dict[str, dict[str, str]] = {}
    for edit, label, sublabel in zip(edits, labels, sublabels, strict=True):
        if not isinstance(label, str) or not isinstance(sublabel, str):
            raise ProtocolError("teacher response labels must be strings")
        raw_label = f"{label}/{sublabel}"
        if raw_label not in allowed:
            raise ProtocolError(f"teacher response contains unknown taxonomy label {raw_label!r}")
        outputs[str(edit["edit_id"])] = {
            "label": label,
            "sublabel": sublabel,
            "raw_label": raw_label,
        }
    return {"labels": labels, "sublabels": sublabels, "outputs_by_edit_id": outputs}
