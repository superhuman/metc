"""Public-source reconstruction dispatch and semantic reference verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .acquire import acquire_sources
from .catalog import RECIPES, dataset_recipe
from .m2 import reconstruct_conll, reconstruct_kolla
from .merlin import reconstruct_merlin
from .mucgec import generate_cherrant_reference, reconstruct_mucgec
from .uagec import reconstruct_uagec


CONLL_COMBINED_SHA256 = "07771f4653844f2732e1d93a5711181995ff2ae0828016a276ad1e7335c96085"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(root: Path, pattern: str, description: str) -> Path:
    if Path(root).is_file():
        paths = [Path(root)]
    else:
        paths = list(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {description} under {root}, found {len(paths)}")
    return paths[0]


def _validate_counts(dataset_id: str, rows: list[dict[str, object]]) -> None:
    recipe = dataset_recipe(dataset_id)
    edits = sum(len(row["gold"]) for row in rows)  # type: ignore[arg-type]
    if len(rows) != recipe.expected_items or edits != recipe.expected_gold_edits:
        raise ValueError(
            f"{dataset_id} reconstruction count mismatch: "
            f"items={len(rows)} (expected {recipe.expected_items}), "
            f"edits={edits} (expected {recipe.expected_gold_edits})"
        )


def reconstruct_references(
    dataset_id: str,
    source_paths: Mapping[str, Path],
    *,
    cherrant_reference: Path | None = None,
) -> list[dict[str, object]]:
    """Reconstruct one paper reference from already acquired public sources."""
    recipe = dataset_recipe(dataset_id)
    missing = set(recipe.source_ids) - set(source_paths)
    if missing:
        raise ValueError(f"{dataset_id} is missing acquired sources: {sorted(missing)}")

    if dataset_id.startswith("en_conll14_"):
        combined = _one(
            source_paths["conll14_test"],
            "official-2014.combined.m2",
            "official combined CoNLL M2",
        )
        if _sha256(combined) != CONLL_COMBINED_SHA256:
            raise ValueError("official combined CoNLL M2 has an unexpected checksum")
        annotator = 0 if dataset_id.endswith("ann0") else 1
        rows = reconstruct_conll(combined, annotator)
    elif dataset_id == "de_merlin":
        rows = reconstruct_merlin(
            source_paths["merlin_exmaralda_v1_1"],
            "german",
            boyd_root=source_paths["boyd_wnut2018_data"],
        )
    elif dataset_id == "cs_merlin":
        rows = reconstruct_merlin(source_paths["merlin_exmaralda_v1_1"], "czech")
    elif dataset_id == "ko_kolla":
        kolla = _one(source_paths["kolla_v2_m2"], "KoLLA_multi-refs.m2", "KoLLA M2")
        rows = reconstruct_kolla(kolla)
    elif dataset_id == "uk_uagec":
        rows = reconstruct_uagec(source_paths["ua_gec"])
    elif dataset_id == "zh_mucgec":
        if cherrant_reference is None:
            raise ValueError("zh_mucgec reconstruction requires a generated ChERRANT M2")
        rows = reconstruct_mucgec(source_paths["mucgec"], cherrant_reference)
    else:  # pragma: no cover - guarded by dataset_recipe
        raise ValueError(f"unsupported dataset: {dataset_id!r}")
    _validate_counts(dataset_id, rows)
    return rows


def reconstruct_from_public_sources(
    dataset_id: str,
    cache_dir: Path,
    *,
    cherrant_python: Path | None = None,
    no_network: bool = False,
) -> list[dict[str, object]]:
    """Acquire and reconstruct one reference, running ChERRANT when required."""
    recipe = dataset_recipe(dataset_id)
    paths = acquire_sources(recipe.source_ids, cache_dir, no_network=no_network)
    cherrant_reference = None
    if dataset_id == "zh_mucgec":
        if cherrant_python is None:
            raise ValueError(
                "zh_mucgec requires --cherrant-python pointing to the pinned Python 3.11 "
                "ChERRANT environment"
            )
        cherrant_reference = generate_cherrant_reference(
            paths["mucgec"], Path(cache_dir) / "generated" / "zh_mucgec", cherrant_python
        )
    return reconstruct_references(dataset_id, paths, cherrant_reference=cherrant_reference)


def reconstruct_all_from_public_sources(
    cache_dir: Path,
    output_dir: Path,
    *,
    cherrant_python: Path,
    no_network: bool = False,
) -> dict[str, Path]:
    """Acquire and reconstruct all seven references into benchmark directories."""
    outputs: dict[str, Path] = {}
    for dataset_id in RECIPES:
        rows = reconstruct_from_public_sources(
            dataset_id,
            cache_dir,
            cherrant_python=cherrant_python if dataset_id == "zh_mucgec" else None,
            no_network=no_network,
        )
        output = Path(output_dir) / dataset_id / "annotations.jsonl"
        write_jsonl(rows, output)
        outputs[dataset_id] = output
    return outputs


def write_jsonl(rows: Iterable[Mapping[str, object]], output: Path) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity(dataset_id: str, row: Mapping[str, object]) -> object:
    return row["sent_idx"] if dataset_id.startswith("en_conll14_") else row["idx"]


def _normalized_gold(dataset_id: str, gold: object) -> object:
    if dataset_id != "uk_uagec":
        return gold
    normalized = []
    for raw_edit in gold:  # type: ignore[union-attr]
        edit = raw_edit  # type: ignore[assignment]
        normalized.append(
            {
                "orig_start": edit["orig_start"],
                "orig_end": edit["orig_end"],
                "raw_label": edit["raw_label"],
                "target": edit["target"],
                "replacement": edit.get("replacement") or "",
            }
        )
    return normalized


def _evaluated_record(dataset_id: str, row: Mapping[str, object]) -> dict[str, object]:
    if dataset_id.startswith("en_conll14_") or dataset_id == "zh_mucgec":
        keys = ("src", "tgt")
    elif dataset_id in {"de_merlin", "cs_merlin"}:
        keys = ("orig_tokens", "th1_tokens")
    else:
        keys = ("orig_tokens", "cor_tokens")
    record = {key: row[key] for key in keys}
    if dataset_id == "cs_merlin":
        record["essay_id"] = row["essay_id"]
    if dataset_id == "uk_uagec":
        record["doc_id"] = row["doc_id"]
        record["para_idx"] = row["para_idx"]
    record["gold"] = _normalized_gold(dataset_id, row["gold"])
    return record


def assert_semantically_equal(
    dataset_id: str,
    reconstructed: Iterable[Mapping[str, object]],
    frozen: Iterable[Mapping[str, object]],
) -> None:
    """Assert equality of every evaluated source, target, span, label, and replacement.

    JSON object key order and the later batched row order are intentionally ignored.
    UA-GEC diagnostic character metadata is ignored because evaluation consumes its
    token spans, labels, and full token-span replacements.
    """
    left = {_identity(dataset_id, row): _evaluated_record(dataset_id, row) for row in reconstructed}
    right = {_identity(dataset_id, row): _evaluated_record(dataset_id, row) for row in frozen}
    if left.keys() != right.keys():
        missing = sorted(set(right) - set(left), key=str)
        extra = sorted(set(left) - set(right), key=str)
        raise AssertionError(
            f"{dataset_id} item identities differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    for identifier in left:
        if left[identifier] != right[identifier]:
            raise AssertionError(f"{dataset_id} differs at item {identifier!r}")


def available_datasets() -> tuple[str, ...]:
    return tuple(RECIPES)
