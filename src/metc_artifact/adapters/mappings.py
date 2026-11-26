"""Exact frozen cross-taxonomy routing used by fresh adapter outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .errors import AdapterOutputError


DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "evaluation" / (
    "cross_taxonomy_mappings.json"
)


@lru_cache(maxsize=4)
def load_mapping_tables(path: Path = DEFAULT_MAPPING_PATH) -> dict[str, dict[Any, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterOutputError(f"cannot load frozen mappings from {path}: {error}") from error
    output: dict[str, dict[Any, str]] = {}
    for mapping in payload.get("mappings", []):
        name = mapping.get("name")
        key_format = mapping.get("key_format")
        if not isinstance(name, str) or key_format not in {"string", "tuple2"}:
            raise AdapterOutputError(f"malformed mapping table in {path}")
        table: dict[Any, str] = {}
        for entry in mapping.get("entries", []):
            raw_key = entry.get("key")
            key = tuple(raw_key) if key_format == "tuple2" else raw_key
            value = entry.get("value")
            if key in table or not isinstance(value, str):
                raise AdapterOutputError(f"malformed or duplicate entry in {name}")
            table[key] = value
        output[name] = table
    return output


def _table(name: str) -> dict[Any, str]:
    try:
        return load_mapping_tables()[name]
    except KeyError as error:
        raise AdapterOutputError(f"frozen mapping table is missing {name}") from error


def map_metc(dataset_id: str, native_label: str) -> str:
    try:
        category, sublabel = native_label.split("/", 1)
    except ValueError as error:
        raise AdapterOutputError(f"invalid METC label: {native_label!r}") from error
    if dataset_id.startswith("en_conll14_"):
        key: Any = (category, sublabel)
        name = "conll.METC_TO_CONLL"
        fallback = None
    elif dataset_id in {"de_merlin", "cs_merlin"}:
        key, name, fallback = native_label, "merlin.METC_TO_MERLIN", "V_semdenot_word_fs"
    elif dataset_id == "ko_kolla":
        key, name, fallback = native_label, "kolla.METC_TO_KOLLA", "R:OTHER"
    elif dataset_id == "uk_uagec":
        key, name, fallback = native_label, "uk.METC_TO_UAGEC", "G/Other"
    elif dataset_id == "zh_mucgec":
        key, name, fallback = native_label, "zh.METC_TO_CHERRANT", "OTHER"
    else:  # pragma: no cover - schema rejects this before mapping
        raise AdapterOutputError(f"METC does not apply to {dataset_id}")
    table = _table(name)
    if key in table:
        return table[key]
    if fallback is not None:
        return fallback
    raise AdapterOutputError(f"unmapped METC label for {dataset_id}: {native_label}")


def map_errant_en(native_label: str, *, unknown_to_others: bool = False) -> str:
    overrides = _table("conll.ERRANT_PREFIX_OVERRIDES")
    if native_label in overrides:
        return overrides[native_label]
    category = native_label[2:] if native_label[:2] in {"M:", "U:", "R:"} else native_label
    table = _table("conll.ERRANT_CATEGORY_TO_CONLL")
    if category in table:
        return table[category]
    if unknown_to_others:
        return "Others"
    raise AdapterOutputError(f"unmapped ERRANT label: {native_label}")


def map_gerrant_de(native_label: str) -> str:
    if native_label in {"UNK", "noop", "NA"}:
        return "V_semdenot_word_fs"
    category = native_label.split(":", 1)[1] if ":" in native_label else native_label
    return _table("merlin.GERRANT_PORT_CAT_TO_MERLIN").get(
        category, "V_semdenot_word_fs"
    )


def _jp_lhs(native_label: str) -> str:
    rest = native_label.split(":", 1)[1] if ":" in native_label else native_label
    lhs = rest.split("->", 1)[0].strip()
    return lhs.split()[0] if lhs else ""


def map_jp(dataset_id: str, native_label: str) -> str:
    if dataset_id.startswith("en_conll14_"):
        return map_errant_en(native_label, unknown_to_others=True)
    if dataset_id in {"de_merlin", "cs_merlin"}:
        return _table("merlin.JP_UPOS_TO_MERLIN").get(
            _jp_lhs(native_label), "V_semdenot_word_fs"
        )
    if dataset_id == "ko_kolla":
        return native_label
    if dataset_id == "uk_uagec":
        if native_label in {"UNK", "noop", "NA"}:
            return "G/Other"
        return _table("uk.JP_UPOS_TO_UAGEC").get(_jp_lhs(native_label), "G/Other")
    if dataset_id == "zh_mucgec":
        return map_jp_zh(native_label)
    raise AdapterOutputError(f"JP-ERRANT does not apply to {dataset_id}")


def map_jp_zh(native_label: str) -> str:
    label = native_label.strip()
    if not label or label.lower() == "noop":
        return ""
    if label in {"W", "WO"}:
        return "W"
    body = label.split(":", 1)[1].strip() if ":" in label else label
    if body.startswith("SPELL"):
        return "SPELL"
    special = _table("zh._JP_SPECIAL")
    if body in special:
        return special[body]
    if "->" in body:
        body = body.split("->", 1)[1]
    head = body.strip().split()[0] if body.strip() else ""
    if head in special:
        return special[head]
    return _table("zh._JP_POS_TO_CAT").get(head, "OTHER")


def map_cherrant_gold(native_label: str) -> str:
    label = native_label.strip()
    if not label or label.lower() == "noop":
        return ""
    if label in {"W", "WO"}:
        return "W"
    category = label.split(":", 1)[1].strip() if ":" in label else label
    category = category.split()[0] if category else category
    allowed = {
        "VERB",
        "NOUN",
        "PUNCT",
        "AUX",
        "ADV",
        "PREP",
        "PRON",
        "CONJ",
        "ADJ",
        "NUM",
        "QUAN",
        "SPELL",
        "OTHER",
        "W",
    }
    return category if category in allowed else "OTHER"
