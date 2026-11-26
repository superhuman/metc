"""Public identifiers derived only from released scientific identity fields."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SAMPLE_ID_VERSION = "sample-v2"


def public_sample_id(dataset_id: str, row_idx: Any, system: str) -> str:
    """Return a stable sample ID without incorporating private platform IDs."""

    serialized = json.dumps(
        [dataset_id, row_idx, system],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{SAMPLE_ID_VERSION}:{digest[:24]}"
