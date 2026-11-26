"""Raw-row judge metrics matching the paper evaluation definitions."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping

from .bootstrap import (
    clustered_counter_bootstrap,
    discrete_interval,
    linear_percentile,
    mean_bootstrap,
)
from .loaders import JudgeMetricsInputError, JudgeVerdictCorpus, QUALITY_LEVELS


QUALITY_ORDER = ("Good", "Acceptable", "Bad")
SYSTEM_ORDER = ("metc", "metc_teacher", "errant", "ger", "jp")
ADEQUACY_SCORE = {"Perfect": 1.0, "Acceptable": 0.5, "Bad": 0.0}


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {
        "dataset",
        "item_id",
        "item_order",
        "edit_id",
        "system",
        "primary_span",
        "primary_label",
        "second_span",
        "second_label",
    }
    for position, raw_row in enumerate(rows, 1):
        row = dict(raw_row)
        missing = required - set(row)
        if missing:
            raise JudgeMetricsInputError(f"judge row {position} lacks fields {sorted(missing)!r}")
        edit_id = row["edit_id"]
        if not isinstance(edit_id, str) or edit_id in seen:
            raise JudgeMetricsInputError(
                f"judge row {position} has an invalid or duplicate edit_id"
            )
        if not isinstance(row["item_order"], int) or isinstance(row["item_order"], bool):
            raise JudgeMetricsInputError(f"judge row {position} has a non-integer item_order")
        for judge in ("primary", "second"):
            for axis in ("span", "label"):
                value = row[f"{judge}_{axis}"]
                if value not in QUALITY_LEVELS:
                    raise JudgeMetricsInputError(
                        f"judge row {position} has invalid {judge}_{axis}={value!r}"
                    )
        seen.add(edit_id)
        output.append(row)
    if not output:
        raise JudgeMetricsInputError("raw judge metrics require at least one edit")
    return output


def _judge_field(judge: str, axis: str) -> str:
    if judge not in {"primary", "second"}:
        raise ValueError("judge must be 'primary' or 'second'")
    if axis not in {"span", "label"}:
        raise ValueError("axis must be 'span' or 'label'")
    return f"{judge}_{axis}"


def summarize_acceptability(
    rows: Iterable[Mapping[str, Any]],
    judge: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute sentence-clustered edit-level span and label quality rates."""

    validated = _validated_rows(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        grouped[(str(row["dataset"]), str(row["system"]))].append(row)

    result: dict[str, Any] = {}
    for (dataset, system), system_rows in sorted(grouped.items()):
        by_item: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in system_rows:
            by_item[(int(row["item_order"]), str(row["item_id"]))].append(row)
        item_keys = sorted(by_item)
        system_result: dict[str, Any] = {}
        for axis in ("span", "label"):
            field = _judge_field(judge, axis)
            clusters = [Counter(row[field] for row in by_item[key]) for key in item_keys]
            overall = Counter(row[field] for row in system_rows)
            distributions = clustered_counter_bootstrap(
                clusters,
                levels=QUALITY_ORDER,
                samples=samples,
                seed=seed,
            )
            total = len(system_rows)
            ratings = {}
            for level in QUALITY_ORDER:
                ratings[level] = {
                    "count": overall[level],
                    "n_total": total,
                    "pct": overall[level] / total * 100,
                    "ci_low": linear_percentile(distributions[level], 0.025),
                    "ci_high": linear_percentile(distributions[level], 0.975),
                }
            system_result[axis] = ratings
        result.setdefault(dataset, {})[system] = system_result
    return result


def sentence_verdict(qualities: Iterable[tuple[str, str]]) -> str:
    """Reduce per-edit span/label verdicts to the paper's sentence verdict."""

    values = list(qualities)
    if not values:
        raise ValueError("sentence adequacy requires at least one edit")
    if any(span == "Bad" or label == "Bad" for span, label in values):
        return "Bad"
    if all(span == "Good" and label == "Good" for span, label in values):
        return "Perfect"
    return "Acceptable"


def summarize_adequacy(
    rows: Iterable[Mapping[str, Any]],
    judge: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute sentence adequacy directly from per-edit judge verdicts."""

    validated = _validated_rows(rows)
    span_field = _judge_field(judge, "span")
    label_field = _judge_field(judge, "label")
    grouped: dict[tuple[str, str, int, str], list[tuple[str, str]]] = defaultdict(list)
    for row in validated:
        grouped[
            (
                str(row["dataset"]),
                str(row["system"]),
                int(row["item_order"]),
                str(row["item_id"]),
            )
        ].append((row[span_field], row[label_field]))

    by_system: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    for (dataset, system, item_order, item_id), qualities in grouped.items():
        by_system[(dataset, system)].append((item_order, item_id, sentence_verdict(qualities)))

    result: dict[str, Any] = {}
    for (dataset, system), values in sorted(by_system.items()):
        verdicts = [value[2] for value in sorted(values)]
        counts = Counter(verdicts)
        scores = [ADEQUACY_SCORE[verdict] for verdict in verdicts]
        bootstrap = [value * 100 for value in mean_bootstrap(scores, samples=samples, seed=seed)]
        low, high = discrete_interval(bootstrap)
        item_count = len(verdicts)
        result.setdefault(dataset, {})[system] = {
            "n_items": item_count,
            "perfect_pct": counts["Perfect"] / item_count * 100,
            "acceptable_pct": counts["Acceptable"] / item_count * 100,
            "bad_pct": counts["Bad"] / item_count * 100,
            "adequacy_pct": sum(scores) / item_count * 100,
            "ci_low": low,
            "ci_high": high,
        }
    return result


def _agreement_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("judge agreement requires at least one edit")
    span_matches = sum(row["primary_span"] == row["second_span"] for row in rows)
    label_matches = sum(row["primary_label"] == row["second_label"] for row in rows)
    primary_good = sum(row["primary_label"] == "Good" for row in rows)
    second_good = sum(row["second_label"] == "Good" for row in rows)
    return {
        "n_edits": count,
        "span_agreement_pct": span_matches / count * 100,
        "label_agreement_pct": label_matches / count * 100,
        "primary_label_good_pct": primary_good / count * 100,
        "second_label_good_pct": second_good / count * 100,
    }


def summarize_judge_agreement(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute exact primary/second agreement on stable edit IDs."""

    validated = _validated_rows(rows)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_system: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        dataset = str(row["dataset"])
        system = str(row["system"])
        by_dataset[dataset].append(row)
        by_system[(dataset, system)].append(row)

    result = {}
    for dataset, dataset_rows in sorted(by_dataset.items()):
        systems = {
            system: _agreement_summary(system_rows)
            for (current_dataset, system), system_rows in sorted(by_system.items())
            if current_dataset == dataset
        }
        result[dataset] = {
            "overall": _agreement_summary(dataset_rows),
            "systems": systems,
        }
    return result


def _student_systems(systems: Iterable[str]) -> list[str]:
    observed = set(systems) - {"gold"}
    ordered = [system for system in SYSTEM_ORDER if system in observed]
    return ordered + sorted(observed - set(ordered))


def _pair_result(
    point: float,
    bootstrap: list[float],
    shared: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    bootstrap.sort()
    low = bootstrap[int(0.025 * samples)]
    high = bootstrap[min(samples - 1, int(0.975 * samples))]
    return {
        "n_shared": shared,
        "difference_points": point,
        "ci_low": low,
        "ci_high": high,
        "samples": samples,
        "seed": seed,
        "zero_in_interval": low <= 0 <= high,
    }


def _paired_adequacy_both(
    scores_a: Mapping[tuple[int, str], float],
    scores_b: Mapping[tuple[int, str], float],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    shared = sorted(set(scores_a).intersection(scores_b))
    if not shared:
        raise ValueError("paired adequacy requires at least one shared item")
    forward = [scores_a[key] - scores_b[key] for key in shared]
    reverse = [scores_b[key] - scores_a[key] for key in shared]
    count = len(shared)
    forward_bootstrap: list[float] = []
    reverse_bootstrap: list[float] = []
    rng = random.Random(seed)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        forward_bootstrap.append(sum(forward[index] for index in indices) / count * 100)
        reverse_bootstrap.append(sum(reverse[index] for index in indices) / count * 100)
    return (
        _pair_result(
            sum(forward) / count * 100,
            forward_bootstrap,
            count,
            samples,
            seed,
        ),
        _pair_result(
            sum(reverse) / count * 100,
            reverse_bootstrap,
            count,
            samples,
            seed,
        ),
    )


def summarize_paired_adequacy(
    rows: Iterable[Mapping[str, Any]],
    judge: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare sentence adequacy for every ordered non-reference system pair."""

    validated = _validated_rows(rows)
    span_field = _judge_field(judge, "span")
    label_field = _judge_field(judge, "label")
    grouped: dict[tuple[str, str, int, str], list[tuple[str, str]]] = defaultdict(list)
    for row in validated:
        grouped[
            (
                str(row["dataset"]),
                str(row["system"]),
                int(row["item_order"]),
                str(row["item_id"]),
            )
        ].append((row[span_field], row[label_field]))

    scores: dict[str, dict[str, dict[tuple[int, str], float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (dataset, system, item_order, item_id), qualities in grouped.items():
        scores[dataset][system][(item_order, item_id)] = ADEQUACY_SCORE[sentence_verdict(qualities)]

    output = {}
    for dataset, system_scores in sorted(scores.items()):
        comparisons_out = {}
        for system_a, system_b in combinations(_student_systems(system_scores), 2):
            forward, reverse = _paired_adequacy_both(
                system_scores[system_a],
                system_scores[system_b],
                samples=samples,
                seed=seed,
            )
            comparisons_out[f"{system_a}_minus_{system_b}"] = forward
            comparisons_out[f"{system_b}_minus_{system_a}"] = reverse
        output[dataset] = {
            "scope": "all_ordered_student_system_pairs",
            "pairing": "shared_benchmark_item",
            "comparisons": comparisons_out,
        }
    return output


def _label_difference(good_a: int, total_a: int, good_b: int, total_b: int) -> float:
    if not total_a or not total_b:
        raise ValueError("paired label-Good sample contains no edits")
    return (good_a / total_a - good_b / total_b) * 100


def _paired_label_both(
    clusters_a: Mapping[tuple[int, str], tuple[int, int]],
    clusters_b: Mapping[tuple[int, str], tuple[int, int]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    shared = sorted(set(clusters_a).intersection(clusters_b))
    if not shared:
        raise ValueError("paired label-Good comparison requires a shared item")
    values_a = [clusters_a[key] for key in shared]
    values_b = [clusters_b[key] for key in shared]
    count = len(shared)
    total_good_a = sum(value[0] for value in values_a)
    total_a = sum(value[1] for value in values_a)
    total_good_b = sum(value[0] for value in values_b)
    total_b = sum(value[1] for value in values_b)
    forward_bootstrap: list[float] = []
    reverse_bootstrap: list[float] = []
    rng = random.Random(seed)
    for _ in range(samples):
        good_a = edits_a = good_b = edits_b = 0
        for _ in range(count):
            index = rng.randrange(count)
            good_a += values_a[index][0]
            edits_a += values_a[index][1]
            good_b += values_b[index][0]
            edits_b += values_b[index][1]
        forward_bootstrap.append(_label_difference(good_a, edits_a, good_b, edits_b))
        reverse_bootstrap.append(_label_difference(good_b, edits_b, good_a, edits_a))
    return (
        _pair_result(
            _label_difference(total_good_a, total_a, total_good_b, total_b),
            forward_bootstrap,
            count,
            samples,
            seed,
        ),
        _pair_result(
            _label_difference(total_good_b, total_b, total_good_a, total_a),
            reverse_bootstrap,
            count,
            samples,
            seed,
        ),
    )


def summarize_paired_label_good(
    rows: Iterable[Mapping[str, Any]],
    judge: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare edit-level label-Good rates on shared benchmark items."""

    validated = _validated_rows(rows)
    label_field = _judge_field(judge, "label")
    item_counts: dict[tuple[str, str, int, str], Counter[str]] = defaultdict(Counter)
    for row in validated:
        item_counts[
            (
                str(row["dataset"]),
                str(row["system"]),
                int(row["item_order"]),
                str(row["item_id"]),
            )
        ][row[label_field]] += 1

    clusters: dict[str, dict[str, dict[tuple[int, str], tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (dataset, system, item_order, item_id), counts in item_counts.items():
        clusters[dataset][system][(item_order, item_id)] = (
            counts["Good"],
            sum(counts.values()),
        )

    output = {}
    for dataset, system_clusters in sorted(clusters.items()):
        comparisons_out = {}
        for system_a, system_b in combinations(_student_systems(system_clusters), 2):
            forward, reverse = _paired_label_both(
                system_clusters[system_a],
                system_clusters[system_b],
                samples=samples,
                seed=seed,
            )
            comparisons_out[f"{system_a}_minus_{system_b}"] = forward
            comparisons_out[f"{system_b}_minus_{system_a}"] = reverse
        output[dataset] = {
            "scope": "all_ordered_student_system_pairs",
            "metric": "edit_level_label_good_pct",
            "pairing": "shared_benchmark_item",
            "comparisons": comparisons_out,
        }
    return output


def compute_judge_metrics(
    corpus: JudgeVerdictCorpus,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Derive every LLM-judge metric from complete joined per-edit verdicts."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if corpus.judge_source not in {"frozen_authoritative", "fresh_rerun"}:
        raise JudgeMetricsInputError(f"unknown judge source provenance {corpus.judge_source!r}")
    rows = _validated_rows(corpus.rows)
    return {
        "schema_version": "metc-raw-judge-metrics-v1",
        "judge_source": corpus.judge_source,
        "n_edits": len(rows),
        "bootstrap": {
            "samples": samples,
            "seed": seed,
            "cluster": "benchmark_item",
        },
        "acceptability": {
            judge: summarize_acceptability(rows, judge, samples=samples, seed=seed)
            for judge in ("primary", "second")
        },
        "sentence_adequacy": {
            judge: summarize_adequacy(rows, judge, samples=samples, seed=seed)
            for judge in ("primary", "second")
        },
        "paired_label_good": {
            judge: summarize_paired_label_good(rows, judge, samples=samples, seed=seed)
            for judge in ("primary", "second")
        },
        "paired_sentence_adequacy": {
            judge: summarize_paired_adequacy(rows, judge, samples=samples, seed=seed)
            for judge in ("primary", "second")
        },
        "judge_agreement": summarize_judge_agreement(rows),
    }
