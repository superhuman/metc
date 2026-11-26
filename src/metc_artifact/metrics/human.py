"""Human-evaluation metrics computed from the released assignment records.

The complete Data artifact is the source of truth.  This module reads the
de-identified assignment exports, their complete sampling frames, and the
canonical per-edit verdicts from both LLM judges.  It does not read any
precomputed human-evaluation statistics.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .judges import ADEQUACY_SCORE, QUALITY_ORDER, sentence_verdict
from .loaders import JudgeMetricsInputError, load_frozen_judge_rows


HUMAN_SOURCE = "frozen_authoritative"
SYSTEM_ORDER = ("gold", "metc", "errant", "ger", "jp")


class HumanMetricsInputError(ValueError):
    """Raised when released human records cannot support the raw analysis."""


@dataclass(frozen=True)
class HumanStudyCorpus:
    """Validated raw records for one released human-evaluation study."""

    dataset_id: str
    annotator_pool_size: int | None
    individual_annotator_ids_available: bool
    passes_per_sample: int
    selection_rule: str
    assignments: tuple[dict[str, Any], ...]
    sampling_frame: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HumanMetricsInputError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise HumanMetricsInputError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise HumanMetricsInputError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as error:
        raise HumanMetricsInputError(f"cannot read {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HumanMetricsInputError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise HumanMetricsInputError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _artifact_path(data_root: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HumanMetricsInputError(f"{context}: missing artifact path")
    path = (data_root / relative).resolve()
    try:
        path.relative_to(data_root)
    except ValueError as error:
        raise HumanMetricsInputError(f"{context}: path escapes the Data artifact") from error
    if not path.is_file():
        raise HumanMetricsInputError(f"{context}: missing file {relative}")
    return path


def _quality(value: Any, context: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if value not in QUALITY_ORDER:
        raise HumanMetricsInputError(
            f"{context}: expected Good, Acceptable, or Bad, got {value!r}"
        )
    return str(value)


def _validate_edit_list(
    edits: Any,
    *,
    context: str,
    qualities_optional: bool,
) -> list[dict[str, Any]]:
    if not isinstance(edits, list) or not edits:
        raise HumanMetricsInputError(f"{context}: edits must be a non-empty list")
    output: list[dict[str, Any]] = []
    seen_slots: set[int] = set()
    seen_ids: set[str] = set()
    for position, raw_edit in enumerate(edits, 1):
        if not isinstance(raw_edit, dict):
            raise HumanMetricsInputError(f"{context}.edits[{position}]: expected an object")
        edit = dict(raw_edit)
        slot = edit.get("slot")
        edit_id = edit.get("edit_id")
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1:
            raise HumanMetricsInputError(f"{context}.edits[{position}]: invalid slot")
        if not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:"):
            raise HumanMetricsInputError(f"{context}.edits[{position}]: invalid stable edit ID")
        if slot in seen_slots or edit_id in seen_ids:
            raise HumanMetricsInputError(f"{context}: duplicate edit slot or stable ID")
        seen_slots.add(slot)
        seen_ids.add(edit_id)
        if not qualities_optional:
            if "span_quality" in edit or "label_quality" in edit:
                raise HumanMetricsInputError(
                    f"{context}.edits[{position}]: sampling frame contains verdicts"
                )
        else:
            edit["span_quality"] = _quality(
                edit.get("span_quality"),
                f"{context}.edits[{position}].span_quality",
                optional=True,
            )
            edit["label_quality"] = _quality(
                edit.get("label_quality"),
                f"{context}.edits[{position}].label_quality",
                optional=True,
            )
            if not isinstance(edit.get("unannotatable"), bool):
                raise HumanMetricsInputError(
                    f"{context}.edits[{position}].unannotatable: expected a boolean"
                )
        output.append(edit)
    output.sort(key=lambda edit: edit["slot"])
    if [edit["slot"] for edit in output] != list(range(1, len(output) + 1)):
        raise HumanMetricsInputError(f"{context}: edit slots are not contiguous")
    return output


def _expected_assignment_status(assignment: Mapping[str, Any]) -> str:
    edits = assignment["edits"]
    complete = all(
        edit["span_quality"] is not None and edit["label_quality"] is not None
        for edit in edits
    )
    any_unannotatable = any(edit["unannotatable"] for edit in edits)
    session_unannotatable = assignment.get("session_unannotatable")
    if not isinstance(session_unannotatable, bool):
        raise HumanMetricsInputError("assignment session_unannotatable must be a boolean")
    if complete and not any_unannotatable and not session_unannotatable:
        return "scorable"
    if any_unannotatable or session_unannotatable:
        return "unannotatable"
    return "incomplete"


def _manifest_int(value: Any, context: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HumanMetricsInputError(f"{context}: expected a non-negative integer")
    return value


def load_human_study(data_root: Path, dataset_id: str) -> HumanStudyCorpus:
    """Load and cross-check one raw released human-evaluation study."""

    data_root = data_root.resolve()
    manifest = _read_json(data_root / "manifest.full.json")
    studies = manifest.get("human_annotations")
    if not isinstance(studies, dict) or dataset_id not in studies:
        raise HumanMetricsInputError(f"Data artifact has no human study for {dataset_id}")
    metadata = studies[dataset_id]
    if not isinstance(metadata, dict):
        raise HumanMetricsInputError(f"human_annotations.{dataset_id}: expected an object")
    frame_metadata = metadata.get("sampling_frame")
    if not isinstance(frame_metadata, dict):
        raise HumanMetricsInputError(
            f"human_annotations.{dataset_id}.sampling_frame: expected an object"
        )

    assignment_path = _artifact_path(
        data_root, metadata.get("path"), f"human_annotations.{dataset_id}"
    )
    frame_path = _artifact_path(
        data_root,
        frame_metadata.get("path"),
        f"human_annotations.{dataset_id}.sampling_frame",
    )
    raw_assignments = _read_jsonl(assignment_path)
    raw_frame = _read_jsonl(frame_path)

    frame_by_sample: dict[str, dict[str, Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    for position, raw_row in enumerate(raw_frame, 1):
        row = dict(raw_row)
        context = f"{frame_path}:{position}"
        if row.get("dataset_id") != dataset_id:
            raise HumanMetricsInputError(f"{context}: wrong dataset_id")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in frame_by_sample:
            raise HumanMetricsInputError(f"{context}: invalid or duplicate sample_id")
        edits = _validate_edit_list(
            row.get("edits"), context=context, qualities_optional=False
        )
        if row.get("edit_count") != len(edits):
            raise HumanMetricsInputError(f"{context}: edit_count does not match edits")
        if not isinstance(row.get("selected_for_human_study"), bool):
            raise HumanMetricsInputError(
                f"{context}: selected_for_human_study must be a boolean"
            )
        if not isinstance(row.get("selection_rank"), int):
            raise HumanMetricsInputError(f"{context}: invalid selection_rank")
        row["edits"] = edits
        frame_rows.append(row)
        frame_by_sample[sample_id] = row
    frame_rows.sort(key=lambda row: row["selection_rank"])

    assignments: list[dict[str, Any]] = []
    assignments_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignment_orders: set[int] = set()
    for position, raw_assignment in enumerate(raw_assignments, 1):
        assignment = dict(raw_assignment)
        context = f"{assignment_path}:{position}"
        if assignment.get("dataset_id") != dataset_id:
            raise HumanMetricsInputError(f"{context}: wrong dataset_id")
        sample_id = assignment.get("sample_id")
        if sample_id not in frame_by_sample:
            raise HumanMetricsInputError(f"{context}: sample is absent from the sampling frame")
        frame = frame_by_sample[sample_id]
        if not frame["selected_for_human_study"]:
            raise HumanMetricsInputError(f"{context}: assignment is outside the selected sample")
        order = assignment.get("assignment_order")
        if (
            not isinstance(order, int)
            or isinstance(order, bool)
            or order < 0
            or order in assignment_orders
        ):
            raise HumanMetricsInputError(f"{context}: invalid or duplicate assignment_order")
        assignment_orders.add(order)
        edits = _validate_edit_list(
            assignment.get("edits"), context=context, qualities_optional=True
        )
        if assignment.get("edit_count") != len(edits):
            raise HumanMetricsInputError(f"{context}: edit_count does not match edits")
        for field in ("item_id", "row_idx", "system"):
            if assignment.get(field) != frame.get(field):
                raise HumanMetricsInputError(f"{context}: {field} differs from sampling frame")
        frame_edits = {
            edit["slot"]: (edit["edit_id"], edit.get("edit_idx"), edit.get("label"))
            for edit in frame["edits"]
        }
        assignment_edits = {
            edit["slot"]: (edit["edit_id"], edit.get("edit_idx"), edit.get("label"))
            for edit in edits
        }
        if assignment_edits != frame_edits:
            raise HumanMetricsInputError(f"{context}: edit identities differ from sampling frame")
        assignment["edits"] = edits
        expected_status = _expected_assignment_status(assignment)
        if assignment.get("status") != expected_status:
            raise HumanMetricsInputError(
                f"{context}: status {assignment.get('status')!r} should be {expected_status!r}"
            )
        assignments.append(assignment)
        assignments_by_sample[str(sample_id)].append(assignment)
    assignments.sort(key=lambda row: row["assignment_order"])

    passes_per_sample = _manifest_int(
        metadata.get("passes_per_sample"),
        f"human_annotations.{dataset_id}.passes_per_sample",
    )
    assert passes_per_sample is not None
    selected_samples = {
        row["sample_id"] for row in frame_rows if row["selected_for_human_study"]
    }
    if set(assignments_by_sample) != selected_samples:
        raise HumanMetricsInputError(
            f"{dataset_id}: assignment samples differ from the selected sampling frame"
        )
    for sample_id, sample_assignments in assignments_by_sample.items():
        if len(sample_assignments) != passes_per_sample:
            raise HumanMetricsInputError(
                f"{dataset_id}/{sample_id}: expected {passes_per_sample} passes"
            )
        pass_ids = {assignment.get("pass_id") for assignment in sample_assignments}
        if len(pass_ids) != passes_per_sample or not all(
            isinstance(pass_id, str) for pass_id in pass_ids
        ):
            raise HumanMetricsInputError(
                f"{dataset_id}/{sample_id}: pass IDs are missing or duplicated"
            )

    expected_assignments = _manifest_int(
        metadata.get("assignments"), f"human_annotations.{dataset_id}.assignments"
    )
    expected_samples = _manifest_int(
        metadata.get("samples"), f"human_annotations.{dataset_id}.samples"
    )
    expected_frame_items = _manifest_int(
        frame_metadata.get("instrument_items"),
        f"human_annotations.{dataset_id}.sampling_frame.instrument_items",
    )
    if len(assignments) != expected_assignments:
        raise HumanMetricsInputError(f"{dataset_id}: assignment count differs from manifest")
    if len(selected_samples) != expected_samples:
        raise HumanMetricsInputError(f"{dataset_id}: sample count differs from manifest")
    if len(frame_rows) != expected_frame_items:
        raise HumanMetricsInputError(f"{dataset_id}: sampling-frame count differs from manifest")

    try:
        judges = load_frozen_judge_rows(data_root)
    except JudgeMetricsInputError as error:
        raise HumanMetricsInputError(f"cannot load canonical judges: {error}") from error
    judge_rows = tuple(row for row in judges.rows if row["dataset"] == dataset_id)
    judge_by_edit = {row["edit_id"]: row for row in judge_rows}
    for frame in frame_rows:
        for edit in frame["edits"]:
            judge = judge_by_edit.get(edit["edit_id"])
            if judge is None:
                raise HumanMetricsInputError(
                    f"{dataset_id}/{frame['sample_id']}: edit is absent from judges"
                )
            if judge["system"] != frame["system"] or judge["label"] != edit["label"]:
                raise HumanMetricsInputError(
                    f"{dataset_id}/{edit['edit_id']}: judge identity differs from frame"
                )

    annotator_pool_size = _manifest_int(
        metadata.get("annotator_pool_size"),
        f"human_annotations.{dataset_id}.annotator_pool_size",
        optional=True,
    )
    individual_ids = metadata.get("individual_annotator_ids_available")
    if not isinstance(individual_ids, bool):
        raise HumanMetricsInputError(
            f"human_annotations.{dataset_id}.individual_annotator_ids_available: "
            "expected a boolean"
        )
    selection_rule = frame_metadata.get("selection_rule")
    if not isinstance(selection_rule, str) or not selection_rule:
        raise HumanMetricsInputError(f"{dataset_id}: missing sampling rule")
    return HumanStudyCorpus(
        dataset_id=dataset_id,
        annotator_pool_size=annotator_pool_size,
        individual_annotator_ids_available=individual_ids,
        passes_per_sample=passes_per_sample,
        selection_rule=selection_rule,
        assignments=tuple(assignments),
        sampling_frame=tuple(frame_rows),
        judge_rows=judge_rows,
    )


def _historical_interval(values: Sequence[float]) -> tuple[float, float]:
    """Return the percentile indices used by the released human-eval analysis."""

    if not values:
        raise ValueError("cannot form an interval from no bootstrap values")
    ordered = sorted(values)
    count = len(ordered)
    return ordered[int(0.025 * count)], ordered[max(0, int(0.975 * count) - 1)]


def _mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("mean bootstrap requires at least one value")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    count = len(values)
    bootstrap = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count * 100
        for _ in range(samples)
    ]
    return _historical_interval(bootstrap)


def _counter_rates(
    clusters: Sequence[Counter[str]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if not clusters:
        raise ValueError("categorical rates require at least one sample cluster")
    overall: Counter[str] = Counter()
    for cluster in clusters:
        overall.update(cluster)
    total = sum(overall.values())
    if not total:
        raise ValueError("categorical rates contain no verdicts")
    rng = random.Random(seed)
    distributions = {quality: [] for quality in QUALITY_ORDER}
    count = len(clusters)
    for _ in range(samples):
        sampled: Counter[str] = Counter()
        for _ in range(count):
            sampled.update(clusters[rng.randrange(count)])
        denominator = sum(sampled.values())
        for quality in QUALITY_ORDER:
            distributions[quality].append(sampled[quality] / denominator * 100)
    return {
        quality: {
            "count": overall[quality],
            "n_total": total,
            "pct": overall[quality] / total * 100,
            "ci_low": _historical_interval(distributions[quality])[0],
            "ci_high": _historical_interval(distributions[quality])[1],
        }
        for quality in QUALITY_ORDER
    }


def _agreement(values: Iterable[tuple[str, str]]) -> dict[str, float | int | None]:
    pairs = list(values)
    count = len(pairs)
    return {
        "n_verdicts": count,
        "agreement_pct": (
            sum(first == second for first, second in pairs) / count * 100
            if count
            else None
        ),
    }


def _both_axis_agreement(
    values: Iterable[tuple[str, str, str, str]],
) -> dict[str, float | int | None]:
    rows = list(values)
    count = len(rows)
    return {
        "n_verdicts": count,
        "span_agreement_pct": (
            sum(left_span == right_span for left_span, _, right_span, _ in rows)
            / count
            * 100
            if count
            else None
        ),
        "label_agreement_pct": (
            sum(left_label == right_label for _, left_label, _, right_label in rows)
            / count
            * 100
            if count
            else None
        ),
    }


def _ordered_systems(systems: Iterable[str]) -> list[str]:
    observed = set(systems)
    ordered = [system for system in SYSTEM_ORDER if system in observed]
    return ordered + sorted(observed - set(ordered))


def _pair_result(
    *,
    system_a: str,
    system_b: str,
    point: float,
    bootstrap: Sequence[float],
    samples: int,
    seed: int,
    resampling_unit: str,
    n_a: int,
    n_b: int,
    n_clusters: int,
    n_shared_clusters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    low, high = _historical_interval(bootstrap)
    common = {
        "samples": samples,
        "seed": seed,
        "resampling_unit": resampling_unit,
        "n_a": n_a,
        "n_b": n_b,
        "n_clusters": n_clusters,
        "n_shared_clusters": n_shared_clusters,
    }
    forward = {
        **common,
        "system_a": system_a,
        "system_b": system_b,
        "difference_points": point,
        "ci_low": low,
        "ci_high": high,
        "zero_in_interval": low <= 0 <= high,
    }
    reverse = {
        **common,
        "system_a": system_b,
        "system_b": system_a,
        "n_a": n_b,
        "n_b": n_a,
        "difference_points": -point,
        "ci_low": -high,
        "ci_high": -low,
        "zero_in_interval": low <= 0 <= high,
    }
    return forward, reverse


def _independent_mean_pair(
    system_a: str,
    values_a: Sequence[float],
    system_b: str,
    values_b: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        mean_a = sum(values_a[rng.randrange(len(values_a))] for _ in values_a) / len(
            values_a
        )
        mean_b = sum(values_b[rng.randrange(len(values_b))] for _ in values_b) / len(
            values_b
        )
        bootstrap.append((mean_a - mean_b) * 100)
    return _pair_result(
        system_a=system_a,
        system_b=system_b,
        point=(sum(values_a) / len(values_a) - sum(values_b) / len(values_b)) * 100,
        bootstrap=bootstrap,
        samples=samples,
        seed=seed,
        resampling_unit="human_sample",
        n_a=len(values_a),
        n_b=len(values_b),
        n_clusters=len(values_a) + len(values_b),
        n_shared_clusters=0,
    )


def _independent_counter_pair(
    system_a: str,
    values_a: Sequence[Counter[str]],
    system_b: str,
    values_b: Sequence[Counter[str]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def rate(values: Sequence[Counter[str]]) -> float:
        good = sum(value["Good"] for value in values)
        total = sum(sum(value.values()) for value in values)
        return good / total * 100

    rng = random.Random(seed)
    bootstrap = []
    for _ in range(samples):
        sampled_a = [values_a[rng.randrange(len(values_a))] for _ in values_a]
        sampled_b = [values_b[rng.randrange(len(values_b))] for _ in values_b]
        bootstrap.append(rate(sampled_a) - rate(sampled_b))
    return _pair_result(
        system_a=system_a,
        system_b=system_b,
        point=rate(values_a) - rate(values_b),
        bootstrap=bootstrap,
        samples=samples,
        seed=seed,
        resampling_unit="human_sample",
        n_a=len(values_a),
        n_b=len(values_b),
        n_clusters=len(values_a) + len(values_b),
        n_shared_clusters=0,
    )


def _clustered_pair(
    system_a: str,
    clusters_a: Mapping[str, Sequence[Any]],
    system_b: str,
    clusters_b: Mapping[str, Sequence[Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resample base benchmark items, keeping repeated system samples together."""

    cluster_ids = sorted(set(clusters_a).union(clusters_b))
    shared = set(clusters_a).intersection(clusters_b)
    if not cluster_ids:
        raise ValueError("clustered comparison requires at least one base item")

    def value(values: Sequence[Any]) -> float:
        if metric == "mean":
            return sum(values) / len(values) * 100
        if metric == "label_good":
            good = sum(counter["Good"] for counter in values)
            total = sum(sum(counter.values()) for counter in values)
            return good / total * 100
        raise ValueError(f"unknown clustered metric {metric!r}")

    observed_a = [entry for key in cluster_ids for entry in clusters_a.get(key, ())]
    observed_b = [entry for key in cluster_ids for entry in clusters_b.get(key, ())]
    point = value(observed_a) - value(observed_b)
    rng = random.Random(seed)
    bootstrap: list[float] = []
    attempts = 0
    maximum_attempts = samples * 100
    while len(bootstrap) < samples:
        attempts += 1
        if attempts > maximum_attempts:
            raise ValueError("cluster bootstrap repeatedly omitted one comparison system")
        picked = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        sampled_a = [entry for key in picked for entry in clusters_a.get(key, ())]
        sampled_b = [entry for key in picked for entry in clusters_b.get(key, ())]
        if not sampled_a or not sampled_b:
            continue
        bootstrap.append(value(sampled_a) - value(sampled_b))
    return _pair_result(
        system_a=system_a,
        system_b=system_b,
        point=point,
        bootstrap=bootstrap,
        samples=samples,
        seed=seed,
        resampling_unit="benchmark_item",
        n_a=len(observed_a),
        n_b=len(observed_b),
        n_clusters=len(cluster_ids),
        n_shared_clusters=len(shared),
    )


def _comparison_document(
    *,
    systems: Sequence[str],
    sample_values: Mapping[str, Mapping[str, Any]],
    row_by_sample: Mapping[str, str],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    independent: dict[str, dict[str, Any]] = {}
    clustered: dict[str, dict[str, Any]] = {}
    for system_a, system_b in combinations(systems, 2):
        values_a = list(sample_values[system_a].values())
        values_b = list(sample_values[system_b].values())
        if metric == "mean":
            forward, reverse = _independent_mean_pair(
                system_a,
                values_a,
                system_b,
                values_b,
                samples=samples,
                seed=seed,
            )
        else:
            forward, reverse = _independent_counter_pair(
                system_a,
                values_a,
                system_b,
                values_b,
                samples=samples,
                seed=seed,
            )
        independent[f"{system_a}_minus_{system_b}"] = forward
        independent[f"{system_b}_minus_{system_a}"] = reverse

        by_row_a: dict[str, list[Any]] = defaultdict(list)
        by_row_b: dict[str, list[Any]] = defaultdict(list)
        for sample_id, value_a in sample_values[system_a].items():
            by_row_a[row_by_sample[sample_id]].append(value_a)
        for sample_id, value_b in sample_values[system_b].items():
            by_row_b[row_by_sample[sample_id]].append(value_b)
        clustered_forward, clustered_reverse = _clustered_pair(
            system_a,
            by_row_a,
            system_b,
            by_row_b,
            metric=metric,
            samples=samples,
            seed=seed,
        )
        clustered[f"{system_a}_minus_{system_b}"] = clustered_forward
        clustered[f"{system_b}_minus_{system_a}"] = clustered_reverse
    return {
        "independent_human_sample": independent,
        "benchmark_item_clustered": clustered,
    }


def compute_human_metrics(
    corpus: HumanStudyCorpus,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Derive human, human--LLM, and system-comparison metrics from raw records."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    assignments = sorted(
        corpus.assignments,
        key=lambda assignment: (
            int(assignment["assignment_order"]),
            str(assignment["sample_id"]),
        ),
    )
    if not assignments:
        raise HumanMetricsInputError("human study has no assignments")
    judges = {row["edit_id"]: row for row in corpus.judge_rows}
    if len(judges) != len(corpus.judge_rows):
        raise HumanMetricsInputError("canonical judge rows contain duplicate stable IDs")

    systems = _ordered_systems(assignment["system"] for assignment in assignments)
    status_counts: Counter[str] = Counter()
    system_status: dict[str, Counter[str]] = defaultdict(Counter)
    system_assignment_counts: Counter[str] = Counter()
    row_by_sample: dict[str, str] = {}
    span_by_sample: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    label_by_sample: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    unannotatable_edits: Counter[str] = Counter()
    reached_edits: Counter[str] = Counter()
    human_edits: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    complete_occurrences: list[tuple[str, str, str, str, str]] = []
    scorable_assignment_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unannotatable_as_bad: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for assignment in assignments:
        system = str(assignment["system"])
        sample_id = str(assignment["sample_id"])
        row_by_sample[sample_id] = json.dumps(
            assignment["row_idx"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        status = str(assignment["status"])
        status_counts[status] += 1
        system_status[system][status] += 1
        system_assignment_counts[system] += 1
        complete_qualities: list[tuple[str, str]] = []
        for edit in assignment["edits"]:
            edit_id = str(edit["edit_id"])
            span = edit["span_quality"]
            label = edit["label_quality"]
            if span is not None or label is not None or edit["unannotatable"]:
                reached_edits[system] += 1
            if edit["unannotatable"]:
                unannotatable_edits[system] += 1
            if span is not None:
                span_by_sample[system][sample_id][span] += 1
            if label is not None:
                label_by_sample[system][sample_id][label] += 1
            if span is not None and label is not None:
                if edit_id not in judges:
                    raise HumanMetricsInputError(
                        f"human verdict {edit_id} is absent from canonical judges"
                    )
                complete_qualities.append((span, label))
                human_edits[(sample_id, edit_id)].append(
                    (str(assignment["pass_id"]), span, label)
                )
                complete_occurrences.append((system, edit_id, span, label, sample_id))
        if status == "scorable":
            score = ADEQUACY_SCORE[sentence_verdict(complete_qualities)]
            scorable_assignment_scores[system][sample_id].append(score)
            unannotatable_as_bad[system][sample_id].append(score)
        elif status == "unannotatable":
            unannotatable_as_bad[system][sample_id].append(0.0)

    human_edit_quality: dict[str, dict[str, Any]] = {}
    for system in systems:
        system_result: dict[str, Any] = {}
        for axis, by_sample in (
            ("span", span_by_sample[system]),
            ("label", label_by_sample[system]),
        ):
            clusters = [
                by_sample[sample_id]
                for sample_id in by_sample
                if sum(by_sample[sample_id].values())
            ]
            system_result[axis] = _counter_rates(
                clusters, samples=samples, seed=seed
            )
        system_result["unannotatable_edit_pct"] = (
            unannotatable_edits[system] / reached_edits[system] * 100
            if reached_edits[system]
            else None
        )
        human_edit_quality[system] = system_result

    double_rows: list[tuple[str, str, str, str, str]] = []
    double_by_system: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    system_by_edit = {row["edit_id"]: str(row["system"]) for row in corpus.judge_rows}
    for (sample_id, edit_id), verdicts in human_edits.items():
        if len(verdicts) == corpus.passes_per_sample and len(
            {verdict[0] for verdict in verdicts}
        ) == corpus.passes_per_sample:
            first, second = verdicts
            double_rows.append(
                (sample_id, first[1], first[2], second[1], second[2])
            )
            double_by_system[system_by_edit[edit_id]].append(
                (first[1], first[2], second[1], second[2])
            )
    inter_annotator = {
        "scope": "complete_edit_verdicts_on_both_independent_passes",
        "annotator_pool_size": corpus.annotator_pool_size,
        "individual_annotator_ids_available": corpus.individual_annotator_ids_available,
        **_both_axis_agreement(
            (left_span, left_label, right_span, right_label)
            for _, left_span, left_label, right_span, right_label in double_rows
        ),
        "systems": {
            system: _both_axis_agreement(double_by_system[system]) for system in systems
        },
    }
    inter_annotator["n_double_scorable_edits"] = inter_annotator.pop("n_verdicts")

    human_vs_judges: dict[str, Any] = {}
    for judge in ("primary", "second"):
        overall = []
        by_system: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        for system, edit_id, human_span, human_label, _ in complete_occurrences:
            row = judges[edit_id]
            comparison = (
                human_span,
                human_label,
                str(row[f"{judge}_span"]),
                str(row[f"{judge}_label"]),
            )
            overall.append(comparison)
            by_system[system].append(comparison)
        human_vs_judges[judge] = {
            **_both_axis_agreement(overall),
            "systems": {
                system: _both_axis_agreement(by_system[system]) for system in systems
            },
        }
        human_vs_judges[judge]["n_human_verdicts"] = human_vs_judges[judge].pop(
            "n_verdicts"
        )

    unique_edit_ids = sorted({occurrence[1] for occurrence in complete_occurrences})
    judge_agreement = _both_axis_agreement(
        (
            str(judges[edit_id]["primary_span"]),
            str(judges[edit_id]["primary_label"]),
            str(judges[edit_id]["second_span"]),
            str(judges[edit_id]["second_label"]),
        )
        for edit_id in unique_edit_ids
    )
    judge_agreement["n_unique_edits"] = judge_agreement.pop("n_verdicts")
    judge_agreement["weighting"] = "once_per_unique_scorable_edit"

    unique_edits_by_system: dict[str, list[str]] = defaultdict(list)
    for edit_id in unique_edit_ids:
        unique_edits_by_system[system_by_edit[edit_id]].append(edit_id)
    judge_label_good_on_human_edits: dict[str, dict[str, Any]] = {}
    for system in systems:
        judge_label_good_on_human_edits[system] = {}
        edit_ids = unique_edits_by_system[system]
        for judge in ("primary", "second"):
            good = sum(judges[edit_id][f"{judge}_label"] == "Good" for edit_id in edit_ids)
            judge_label_good_on_human_edits[system][judge] = {
                "n_edits": len(edit_ids),
                "label_good_pct": good / len(edit_ids) * 100 if edit_ids else None,
            }

    item_scores: dict[str, dict[str, float]] = defaultdict(dict)
    sentence_adequacy: dict[str, Any] = {}
    two_pass_sensitivity: dict[str, Any] = {}
    unannotatable_sensitivity: dict[str, Any] = {}
    for system in systems:
        ordered_sample_ids = list(scorable_assignment_scores[system])
        item_scores[system] = {
            sample_id: (
                sum(scorable_assignment_scores[system][sample_id])
                / len(scorable_assignment_scores[system][sample_id])
            )
            for sample_id in ordered_sample_ids
        }
        values = list(item_scores[system].values())
        low, high = _mean_interval(values, samples=samples, seed=seed)
        sentence_adequacy[system] = {
            "n_items": len(values),
            "n_scorable_assignments": sum(
                len(scorable_assignment_scores[system][sample_id])
                for sample_id in ordered_sample_ids
            ),
            "adequacy_pct": sum(values) / len(values) * 100,
            "ci_low": low,
            "ci_high": high,
            "ci_half_width": (high - low) / 2,
        }

        double_values = [
            sum(scores) / corpus.passes_per_sample
            for sample_id, scores in scorable_assignment_scores[system].items()
            if len(scores) == corpus.passes_per_sample
        ]
        double_low, double_high = _mean_interval(
            double_values, samples=samples, seed=seed
        )
        two_pass_sensitivity[system] = {
            "n_items": len(double_values),
            "adequacy_pct": sum(double_values) / len(double_values) * 100,
            "ci_low": double_low,
            "ci_high": double_high,
        }

        bad_sample_ids = list(unannotatable_as_bad[system])
        bad_values = [
            sum(unannotatable_as_bad[system][sample_id])
            / len(unannotatable_as_bad[system][sample_id])
            for sample_id in bad_sample_ids
        ]
        bad_low, bad_high = _mean_interval(bad_values, samples=samples, seed=seed)
        unannotatable_sensitivity[system] = {
            "n_items": len(bad_values),
            "n_passes": sum(
                len(unannotatable_as_bad[system][sample_id])
                for sample_id in bad_sample_ids
            ),
            "adequacy_pct": sum(bad_values) / len(bad_values) * 100,
            "ci_low": bad_low,
            "ci_high": bad_high,
        }

    selected_frame = [
        row for row in corpus.sampling_frame if row["selected_for_human_study"]
    ]
    selected_base_items: Counter[str] = Counter(
        json.dumps(
            row["row_idx"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in selected_frame
    )
    repeated_base_items = sum(count > 1 for count in selected_base_items.values())
    excess_samples = sum(count - 1 for count in selected_base_items.values())

    label_sample_values = {
        system: {
            sample_id: counter
            for sample_id, counter in label_by_sample[system].items()
            if sum(counter.values())
        }
        for system in systems
    }
    system_comparisons = {
        "label_good": _comparison_document(
            systems=systems,
            sample_values=label_sample_values,
            row_by_sample=row_by_sample,
            metric="label_good",
            samples=samples,
            seed=seed,
        ),
        "sentence_adequacy": _comparison_document(
            systems=systems,
            sample_values=item_scores,
            row_by_sample=row_by_sample,
            metric="mean",
            samples=samples,
            seed=seed,
        ),
    }

    return {
        "schema_version": "metc-raw-human-metrics-v1",
        "human_source": HUMAN_SOURCE,
        "judge_source": "frozen_authoritative",
        "dataset": corpus.dataset_id,
        "study": {
            "sampling_rule": corpus.selection_rule,
            "sampling_frame_items": len(corpus.sampling_frame),
            "selected_samples": len(selected_frame),
            "assignments": len(assignments),
            "passes_per_sample": corpus.passes_per_sample,
            "annotator_pool_size": corpus.annotator_pool_size,
            "individual_annotator_ids_available": (
                corpus.individual_annotator_ids_available
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "system_assignment_counts": dict(sorted(system_assignment_counts.items())),
            "system_status_counts": {
                system: dict(sorted(system_status[system].items())) for system in systems
            },
            "unique_benchmark_items": len(selected_base_items),
            "repeated_benchmark_items": repeated_base_items,
            "excess_samples_from_repeated_items": excess_samples,
        },
        "bootstrap": {
            "samples": samples,
            "seed": seed,
            "primary_unit": "human_sample",
            "repeated_item_sensitivity_unit": "benchmark_item",
        },
        "human_edit_quality": human_edit_quality,
        "inter_annotator_agreement": inter_annotator,
        "human_vs_judges": human_vs_judges,
        "judge_vs_judge_on_human_edits": judge_agreement,
        "judge_label_good_on_human_edits": judge_label_good_on_human_edits,
        "sentence_adequacy": sentence_adequacy,
        "adequacy_sensitivity": {
            "two_scorable_passes_only": two_pass_sensitivity,
            "unannotatable_pass_as_bad": unannotatable_sensitivity,
        },
        "system_comparisons": system_comparisons,
    }
