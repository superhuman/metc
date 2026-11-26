from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MAPPING_SHA256 = "a65dd2254df04db13db0aeeda9844bca9da1a5645a4d5f79061306fe554bbad8"
EXPECTED_GLOSS_SHA256 = "24ec79277ec9a98b9c1c097be432a31e2a9e16e9499d2a1bd4ea05c8b44923ca"
EXPECTED_GLOSS_COUNT = 3_616


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def test_cross_taxonomy_mappings_match_frozen_digest() -> None:
    payload = json.loads(
        (ROOT / "evaluation" / "cross_taxonomy_mappings.json").read_text(encoding="utf-8")
    )
    digest_input = []
    for mapping in payload["mappings"]:
        entries = []
        for entry in mapping["entries"]:
            key = entry["key"]
            if mapping["key_format"] == "tuple2":
                key_repr = repr(tuple(key))
            else:
                assert mapping["key_format"] == "string"
                key_repr = repr(key)
            entries.append((key_repr, entry["value"]))
        digest_input.append((mapping["name"], sorted(entries)))

    assert payload["frozen_sha256"] == EXPECTED_MAPPING_SHA256
    assert _sha256_json(digest_input) == EXPECTED_MAPPING_SHA256


def test_label_glosses_match_frozen_digest_and_count() -> None:
    payload = json.loads((ROOT / "evaluation" / "label_glosses.json").read_text(encoding="utf-8"))
    entries = payload["entries"]

    assert payload["entry_count"] == EXPECTED_GLOSS_COUNT == len(entries)
    assert payload["entries_sha256"] == EXPECTED_GLOSS_SHA256
    assert _sha256_json(entries) == EXPECTED_GLOSS_SHA256
    keys = [(entry["dataset"], entry["system"], entry["label"]) for entry in entries]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))
