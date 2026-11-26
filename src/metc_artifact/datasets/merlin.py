"""MERLIN EXMARaLDA parser and German/Czech reference reconstruction."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


_AMP_NOT_ENTITY = re.compile(r"&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)")
_IGNORED_TIERS = {
    "tok",
    "TH1",
    "TH1Diff",
    "TH1Spec",
    "TH2",
    "TH2Diff",
    "sentence",
    "markable_scheme",
    "tunit",
    "complextunit",
    "learner",
    "tok_lemma",
    "tok_lemma_bohnet",
    "tok_morph_bohnet",
    "tok_pos",
    "tok_pos_bohnet",
    "tok_pos_stanford",
    "tok_ambiguous_alternative1",
    "tok_ambiguous_alternative2",
    "tok_unreadable",
    "anonymized",
    "citation",
    "closing",
    "comment",
    "deletion",
    "emphasis",
    "entity",
    "error",
    "foreign_word",
    "greeting",
    "image",
    "insertion",
    "repetition",
}

BOYD_DEVELOPMENT_IDS_SHA256 = "f0aea607f108816b0cef41d250c33e4bccefe5bfb66aeb69007ad4ffa3e49b27"


@dataclass(frozen=True)
class MerlinEdit:
    orig_span: tuple[int, int]
    th1_span: tuple[int, int]
    category: str
    sub_spec: str | None


@dataclass(frozen=True)
class MerlinSentence:
    essay_id: str
    sent_idx: int
    orig_tokens: tuple[str, ...]
    th1_tokens: tuple[str, ...]
    edits: tuple[MerlinEdit, ...]


def _timeline_index(identifier: str) -> int:
    if not identifier.startswith("T"):
        raise ValueError(f"unexpected MERLIN timeline identifier: {identifier!r}")
    return int(identifier[1:])


def _load_tiers(root: ET.Element) -> dict[str, list[tuple[int, int, str]]]:
    tiers: dict[str, list[tuple[int, int, str]]] = {}
    for tier in root.iter("tier"):
        category = tier.attrib.get("category", "")
        tiers[category] = [
            (
                _timeline_index(event.attrib["start"]),
                _timeline_index(event.attrib["end"]),
                event.text or "",
            )
            for event in tier.findall("event")
        ]
    return tiers


def _subtier(category: str, tier_categories: list[str]) -> str | None:
    prefix = f"{category}_"
    return next(
        (
            candidate
            for candidate in tier_categories
            if candidate.startswith(prefix) and candidate not in _IGNORED_TIERS
        ),
        None,
    )


def _events_to_token_span(
    events: list[tuple[int, int, str]], start: int, end: int
) -> tuple[int, int]:
    matching = [
        index
        for index, (event_start, event_end, _value) in enumerate(events)
        if event_start >= start and event_end <= end
    ]
    if matching:
        return matching[0], matching[-1] + 1
    insertion = sum(1 for _event_start, event_end, _value in events if event_end <= start)
    return insertion, insertion


def parse_exb(path: Path) -> list[MerlinSentence]:
    raw = Path(path).read_text(encoding="utf-8")
    root = ET.fromstring(_AMP_NOT_ENTITY.sub("&amp;", raw))
    tiers = _load_tiers(root)
    original_events = sorted(tiers.get("tok", ()))
    th1_events = sorted(tiers.get("TH1", ()))
    sentence_events = sorted(tiers.get("sentence", ()))
    markable_events = tiers.get("markable_scheme", ())
    tier_categories = list(tiers)

    output: list[MerlinSentence] = []
    for sent_idx, (sentence_start, sentence_end, _value) in enumerate(sentence_events):
        original = [
            event
            for event in original_events
            if event[0] >= sentence_start and event[1] <= sentence_end
        ]
        th1 = [
            event for event in th1_events if event[0] >= sentence_start and event[1] <= sentence_end
        ]
        edits: list[MerlinEdit] = []
        for start, end, category in markable_events:
            if start < sentence_start or end > sentence_end:
                continue
            sub_spec = None
            sub_tier = _subtier(category, tier_categories)
            if sub_tier is not None:
                sub_spec = next(
                    (
                        value
                        for sub_start, sub_end, value in tiers[sub_tier]
                        if sub_start == start and sub_end == end
                    ),
                    None,
                )
            edits.append(
                MerlinEdit(
                    _events_to_token_span(original, start, end),
                    _events_to_token_span(th1, start, end),
                    category,
                    sub_spec,
                )
            )
        output.append(
            MerlinSentence(
                Path(path).stem,
                sent_idx,
                tuple(value for _start, _end, value in original),
                tuple(value for _start, _end, value in th1),
                tuple(edits),
            )
        )
    return output


def _minimize_edit(sentence: MerlinSentence, edit: MerlinEdit) -> dict[str, object] | None:
    orig_start, orig_end = edit.orig_span
    th1_start, th1_end = edit.th1_span
    while (
        orig_start < orig_end
        and th1_start < th1_end
        and sentence.orig_tokens[orig_start] == sentence.th1_tokens[th1_start]
    ):
        orig_start += 1
        th1_start += 1
    while (
        orig_start < orig_end
        and th1_start < th1_end
        and sentence.orig_tokens[orig_end - 1] == sentence.th1_tokens[th1_end - 1]
    ):
        orig_end -= 1
        th1_end -= 1
    if orig_start == orig_end and th1_start == th1_end:
        return None
    raw_label = edit.category
    if edit.sub_spec:
        raw_label = f"{raw_label}/{edit.sub_spec}"
    return {
        "orig_start": orig_start,
        "orig_end": orig_end,
        "th1_start": th1_start,
        "th1_end": th1_end,
        "raw_label": raw_label,
        "target": edit.category,
    }


def _language_directory(root: Path, language: str) -> Path:
    candidates = sorted(
        path for path in Path(root).rglob(language) if path.is_dir() and any(path.glob("*.exb"))
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one MERLIN {language!r} EXMARaLDA directory under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _selection_file(root: Path) -> Path:
    candidates = list(Path(root).rglob("merlin-id-ctok-zh1.txt.dev"))
    if len(candidates) != 1:
        raise ValueError(f"expected one Boyd MERLIN development ID file under {root}")
    path = candidates[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != BOYD_DEVELOPMENT_IDS_SHA256:
        raise ValueError(
            f"Boyd development ID checksum mismatch: expected "
            f"{BOYD_DEVELOPMENT_IDS_SHA256}, got {actual}"
        )
    return path


def german_development_ids(boyd_root: Path) -> list[str]:
    """Return the 103 public German MERLIN development essay IDs."""
    identifiers = {
        line.split()[0]
        for line in _selection_file(boyd_root).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(identifiers)


def reconstruct_merlin(
    merlin_root: Path, language: str, *, boyd_root: Path | None = None
) -> list[dict[str, object]]:
    if language not in {"german", "czech"}:
        raise ValueError(f"unsupported MERLIN language: {language!r}")
    language_dir = _language_directory(merlin_root, language)
    if language == "german":
        if boyd_root is None:
            raise ValueError("German MERLIN reconstruction requires the Boyd development IDs")
        paths = [
            language_dir / f"{identifier}.exb" for identifier in german_development_ids(boyd_root)
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"MERLIN is missing {len(missing)} selected German essays")
    else:
        paths = sorted(language_dir.glob("*.exb"))

    rows: list[dict[str, object]] = []
    for path in paths:
        for sentence in parse_exb(path):
            if language == "czech" and (not sentence.orig_tokens or not sentence.th1_tokens):
                continue
            gold = [
                minimized
                for edit in sentence.edits
                if (minimized := _minimize_edit(sentence, edit)) is not None
            ]
            row: dict[str, object] = {
                "idx": len(rows),
                "orig_tokens": list(sentence.orig_tokens),
                "th1_tokens": list(sentence.th1_tokens),
                "gold": gold,
            }
            if language == "czech":
                row["essay_id"] = sentence.essay_id
                row = {
                    "idx": row["idx"],
                    "essay_id": row["essay_id"],
                    "orig_tokens": row["orig_tokens"],
                    "th1_tokens": row["th1_tokens"],
                    "gold": row["gold"],
                }
            rows.append(row)
    return rows
