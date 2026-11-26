"""Strict loaders for frozen and freshly rerun per-edit judge verdicts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from ..schema import dataset_spec


JudgeSource = Literal["frozen_authoritative", "fresh_rerun"]
QUALITY_LEVELS = frozenset({"Good", "Acceptable", "Bad"})


class JudgeMetricsInputError(ValueError):
    """Raised when judge rows cannot support an exact raw-row analysis."""


@dataclass(frozen=True)
class JudgeVerdictCorpus:
    """Joined primary/second verdicts with an explicit provenance marker."""

    judge_source: JudgeSource
    rows: tuple[dict[str, Any], ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as error:
        raise JudgeMetricsInputError(f"cannot read {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise JudgeMetricsInputError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise JudgeMetricsInputError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _canonical_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sort_key(value: Any) -> tuple[str, Any]:
    """Match the historical judge export's ordering of heterogeneous IDs."""

    if isinstance(value, (int, str)):
        return type(value).__qualname__, value
    return type(value).__qualname__, _canonical_key(value)


def _benchmark_order(data_root: Path, dataset: str) -> dict[str, tuple[int, str]]:
    path = data_root / "benchmarks" / dataset / "annotations.jsonl"
    raw_ids: list[Any] = []
    seen: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(path), 1):
        if "idx" in row:
            raw_item_id = row["idx"]
        elif "sent_idx" in row:
            raw_item_id = row["sent_idx"]
        else:
            raise JudgeMetricsInputError(f"{path}:{line_number}: missing item ID")
        item_id = _canonical_key(raw_item_id)
        if item_id in seen:
            raise JudgeMetricsInputError(f"{path}: duplicate item ID {item_id}")
        seen.add(item_id)
        raw_ids.append(raw_item_id)
    return {
        _canonical_key(raw_item_id): (item_order, _canonical_key(raw_item_id))
        for item_order, raw_item_id in enumerate(sorted(raw_ids, key=_canonical_sort_key))
    }


def _validate_quality(value: Any, context: str) -> str:
    if value not in QUALITY_LEVELS:
        raise JudgeMetricsInputError(f"{context}: expected Good, Acceptable, or Bad, got {value!r}")
    return str(value)


def _index_stream(
    rows: list[dict[str, Any]], *, dataset: str, judge: str, context: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        edit_id = row.get("edit_id")
        if not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:"):
            raise JudgeMetricsInputError(f"{context}:{line_number}: missing stable edit ID")
        if edit_id in indexed:
            raise JudgeMetricsInputError(f"{context}: duplicate stable edit ID {edit_id}")
        if row.get("errored") not in (None, False):
            raise JudgeMetricsInputError(f"{context}:{line_number}: errored judge row")
        normalized = dict(row)
        normalized["dataset"] = dataset
        normalized["span_quality"] = _validate_quality(
            row.get("span_quality"), f"{context}:{line_number}.span_quality"
        )
        normalized["label_quality"] = _validate_quality(
            row.get("label_quality"), f"{context}:{line_number}.label_quality"
        )
        normalized["judge_role"] = judge
        indexed[edit_id] = normalized
    return indexed


_IDENTITY_FIELDS = (
    "dataset",
    "row_idx",
    "system",
    "edit_idx",
    "src",
    "tgt",
    "label",
    "edit_start",
    "edit_end",
    "edit_diff",
)
_STABLE_EDIT_IDENTITY_FIELDS = tuple(
    field for field in _IDENTITY_FIELDS if field != "label"
)


def _join_streams(
    primary: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    *,
    item_orders: Mapping[str, Mapping[str, tuple[int, str]]],
) -> tuple[dict[str, Any], ...]:
    primary_ids = set(primary)
    second_ids = set(second)
    if primary_ids != second_ids:
        missing = sorted(primary_ids - second_ids)
        extra = sorted(second_ids - primary_ids)
        raise JudgeMetricsInputError(
            "primary and second judge edit universes differ: "
            f"missing_second={missing[:3]!r}, extra_second={extra[:3]!r}"
        )

    joined: list[dict[str, Any]] = []
    for edit_id in sorted(primary_ids):
        first = primary[edit_id]
        other = second[edit_id]
        for field in _IDENTITY_FIELDS:
            if first.get(field) != other.get(field):
                raise JudgeMetricsInputError(
                    f"{edit_id}: primary/second identity differs for {field}"
                )
        dataset = str(first["dataset"])
        row_idx = first.get("row_idx")
        item_key = _canonical_key(row_idx)
        try:
            item_order, item_id = item_orders[dataset][item_key]
        except KeyError as error:
            raise JudgeMetricsInputError(
                f"{edit_id}: row_idx {row_idx!r} is absent from {dataset}"
            ) from error
        joined.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "item_order": item_order,
                "row_idx": row_idx,
                "edit_id": edit_id,
                "edit_idx": first.get("edit_idx"),
                "system": first.get("system"),
                "label": first.get("label"),
                "primary_span": first["span_quality"],
                "primary_label": first["label_quality"],
                "second_span": other["span_quality"],
                "second_label": other["label_quality"],
                "primary_judge": first.get("judge"),
                "second_judge": other.get("judge"),
            }
        )
    joined.sort(
        key=lambda row: (
            row["dataset"],
            row["item_order"],
            row["system"],
            row["edit_idx"],
            row["edit_id"],
        )
    )
    return tuple(joined)


def load_frozen_judge_rows(data_root: Path) -> JudgeVerdictCorpus:
    """Load and strictly join the authoritative canonical judge files."""

    canonical = data_root / "judges" / "canonical"
    primary_files = {path.stem: path for path in sorted((canonical / "primary").glob("*.jsonl"))}
    second_files = {path.stem: path for path in sorted((canonical / "second").glob("*.jsonl"))}
    if not primary_files or set(primary_files) != set(second_files):
        raise JudgeMetricsInputError(
            "canonical primary/second judge dataset files are missing or inconsistent"
        )

    primary: dict[str, dict[str, Any]] = {}
    second: dict[str, dict[str, Any]] = {}
    orders: dict[str, dict[str, tuple[int, str]]] = {}
    for dataset in sorted(primary_files):
        orders[dataset] = _benchmark_order(data_root, dataset)
        for target, path, judge in (
            (primary, primary_files[dataset], "primary"),
            (second, second_files[dataset], "second"),
        ):
            stream = _index_stream(
                _read_jsonl(path), dataset=dataset, judge=judge, context=str(path)
            )
            duplicates = set(target).intersection(stream)
            if duplicates:
                raise JudgeMetricsInputError(
                    f"stable edit IDs occur in multiple datasets: {sorted(duplicates)[:3]!r}"
                )
            target.update(stream)
    return JudgeVerdictCorpus(
        judge_source="frozen_authoritative",
        rows=_join_streams(primary, second, item_orders=orders),
    )


def _fresh_files(run_root: Path, judge: str) -> list[Path]:
    direct = run_root / f"{judge}.jsonl"
    if direct.is_file():
        return [direct]
    directory = run_root / judge
    if directory.is_dir():
        files = sorted(directory.glob("*.jsonl"))
        if files:
            return files
    batch_files = sorted(run_root.glob(f"acceptability-{judge}-*.jsonl"))
    if batch_files:
        return batch_files
    raise JudgeMetricsInputError(f"fresh run lacks {judge} judge JSONL under {run_root}")


def _dataset_from_edit(edit_id: str, input_row: Mapping[str, Any]) -> str:
    declared = input_row.get("dataset_id", input_row.get("dataset"))
    parts = edit_id.split(":")
    if len(parts) != 5:
        raise JudgeMetricsInputError(f"malformed stable edit ID {edit_id!r}")
    inferred = dataset_spec(parts[1]).dataset_id
    if declared is not None and dataset_spec(str(declared)).dataset_id != inferred:
        raise JudgeMetricsInputError(
            f"{edit_id}: declared dataset {declared!r} differs from stable ID"
        )
    return inferred


def _fresh_stream(paths: list[Path], judge: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    failed_attempt_ids: set[str] = set()
    expected_provider = {"primary": "openai", "second": "google"}[judge]
    for path in paths:
        for line_number, record in enumerate(_read_jsonl(path), 1):
            context = f"{path}:{line_number}"
            if record.get("schema_version") != "metc-fresh-rerun-v1":
                raise JudgeMetricsInputError(f"{context}: unsupported fresh schema")
            if record.get("source") != "fresh_rerun":
                raise JudgeMetricsInputError(f"{context}: not a fresh rerun record")
            if record.get("run_kind") != "acceptability":
                raise JudgeMetricsInputError(f"{context}: not an acceptability rerun")
            if record.get("provider") != expected_provider:
                raise JudgeMetricsInputError(
                    f"{context}: {judge} stream must use {expected_provider}"
                )
            stable_ids = record.get("stable_edit_ids")
            outputs = record.get("outputs_by_edit_id")
            input_row = record.get("input")
            if (
                not isinstance(stable_ids, list)
                or not isinstance(outputs, Mapping)
                or not isinstance(input_row, Mapping)
            ):
                raise JudgeMetricsInputError(
                    f"{context}: stable IDs, inputs, or fresh outputs are malformed"
                )
            if any(
                not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:")
                for edit_id in stable_ids
            ):
                raise JudgeMetricsInputError(f"{context}: invalid stable edit ID")
            if record.get("error") is not None:
                if outputs:
                    raise JudgeMetricsInputError(
                        f"{context}: failed attempt unexpectedly contains outputs"
                    )
                failed_attempt_ids.update(stable_ids)
                continue
            if set(stable_ids) != set(outputs):
                raise JudgeMetricsInputError(
                    f"{context}: successful stable IDs and fresh outputs are inconsistent"
                )
            for edit_id in stable_ids:
                if not isinstance(edit_id, str) or edit_id in indexed:
                    raise JudgeMetricsInputError(
                        f"{context}: invalid or duplicate stable edit ID {edit_id!r}"
                    )
                output = outputs[edit_id]
                if not isinstance(output, Mapping):
                    raise JudgeMetricsInputError(f"{context}: fresh output is not an object")
                dataset = _dataset_from_edit(edit_id, input_row)
                row = dict(input_row)
                if row.get("edit_id") not in (None, edit_id):
                    raise JudgeMetricsInputError(
                        f"{context}: input edit ID differs from output edit ID"
                    )
                row.update(output)
                row.update(
                    {
                        "edit_id": edit_id,
                        "dataset": dataset,
                        "judge": f"{record.get('provider')}:{record.get('model')}",
                        "errored": False,
                    }
                )
                indexed.update(_index_stream([row], dataset=dataset, judge=judge, context=context))
    unresolved = failed_attempt_ids - set(indexed)
    if unresolved:
        raise JudgeMetricsInputError(
            f"fresh {judge} run has {len(unresolved)} edits with no successful retry: "
            f"{sorted(unresolved)[:3]!r}"
        )
    return indexed


def load_fresh_judge_rows(data_root: Path, run_root: Path) -> JudgeVerdictCorpus:
    """Load a complete fresh primary/second rerun without replacing frozen rows."""

    primary = _fresh_stream(_fresh_files(run_root, "primary"), "primary")
    second = _fresh_stream(_fresh_files(run_root, "second"), "second")
    canonical_primary = data_root / "judges" / "canonical" / "primary"
    canonical_files = sorted(canonical_primary.glob("*.jsonl"))
    if not canonical_files:
        raise JudgeMetricsInputError(
            "complete Data artifact has no authoritative primary judge universe"
        )
    authoritative: dict[str, dict[str, Any]] = {}
    for path in canonical_files:
        dataset = path.stem
        stream = _index_stream(
            _read_jsonl(path),
            dataset=dataset,
            judge="primary",
            context=str(path),
        )
        duplicates = set(authoritative).intersection(stream)
        if duplicates:
            raise JudgeMetricsInputError(
                "authoritative judge universe contains duplicate IDs: "
                f"{sorted(duplicates)[:3]!r}"
            )
        authoritative.update(stream)
    expected_ids = set(authoritative)
    for judge, stream in (("primary", primary), ("second", second)):
        observed_ids = set(stream)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise JudgeMetricsInputError(
                f"fresh {judge} rerun is not the complete authoritative job universe: "
                f"missing={missing[:3]!r} ({len(missing)} total), "
                f"extra={extra[:3]!r} ({len(extra)} total)"
            )
        for edit_id in sorted(expected_ids):
            for field in _STABLE_EDIT_IDENTITY_FIELDS:
                if stream[edit_id].get(field) != authoritative[edit_id].get(field):
                    raise JudgeMetricsInputError(
                        f"{edit_id}: fresh {judge} stable-ID identity differs from "
                        f"authoritative {field}"
                    )
    datasets = sorted({str(row["dataset"]) for row in (*primary.values(), *second.values())})
    orders = {dataset: _benchmark_order(data_root, dataset) for dataset in datasets}
    return JudgeVerdictCorpus(
        judge_source="fresh_rerun",
        rows=_join_streams(primary, second, item_orders=orders),
    )


def load_judge_rows(
    data_root: Path,
    *,
    judge_source: JudgeSource,
    fresh_run_root: Path | None = None,
) -> JudgeVerdictCorpus:
    """Load the explicitly selected authoritative or fresh judge source."""

    if judge_source == "frozen_authoritative":
        if fresh_run_root is not None:
            raise JudgeMetricsInputError(
                "fresh_run_root must not be set for frozen_authoritative metrics"
            )
        return load_frozen_judge_rows(data_root)
    if judge_source == "fresh_rerun":
        if fresh_run_root is None:
            raise JudgeMetricsInputError("fresh_rerun requires fresh_run_root")
        return load_fresh_judge_rows(data_root, fresh_run_root)
    raise JudgeMetricsInputError(f"unknown judge source {judge_source!r}")
