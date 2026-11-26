"""Frozen mapping between the released classifier IDs and METC labels."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .errors import CheckpointContractError

EXPECTED_LABEL_COUNT = 42
EXPECTED_MAPPING_SHA256 = "48703bafe33bebfde52eaa79d72a1ec7e5af0d5bbd9569bbcedd10f0d974359f"


def mapping_path() -> Path:
    return Path(str(files("metc.resources").joinpath("label_mapping.json")))


def load_label_mapping() -> dict[int, tuple[str, str]]:
    path = mapping_path()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_MAPPING_SHA256:
        raise CheckpointContractError(
            f"label mapping checksum mismatch: {digest} != {EXPECTED_MAPPING_SHA256}"
        )
    forward = json.loads(raw)
    if len(forward) != EXPECTED_LABEL_COUNT:
        raise CheckpointContractError(
            f"expected {EXPECTED_LABEL_COUNT} labels, found {len(forward)}"
        )
    ids = set(forward.values())
    if ids != set(range(EXPECTED_LABEL_COUNT)):
        raise CheckpointContractError("label IDs must be the dense range 0..41")
    reverse: dict[int, tuple[str, str]] = {}
    for qualified_label, label_id in forward.items():
        label, sublabel = qualified_label.split("/", 1)
        reverse[int(label_id)] = (label, sublabel)
    return reverse
