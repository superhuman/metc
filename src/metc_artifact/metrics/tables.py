"""Flat paper-table rows derived from raw judge metrics."""

from __future__ import annotations

from typing import Any, Mapping

from .judges import compute_judge_metrics
from .loaders import JudgeMetricsInputError, JudgeVerdictCorpus


def _source(metrics: Mapping[str, Any]) -> str:
    source = metrics.get("judge_source")
    if source not in {"frozen_authoritative", "fresh_rerun"}:
        raise JudgeMetricsInputError("metrics lack a valid judge_source provenance marker")
    return str(source)


def paper_table_rows(metrics: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten a raw metrics document into deterministic table-ready records."""

    source = _source(metrics)

    edit_quality = []
    for judge, judge_metrics in sorted(metrics["acceptability"].items()):
        for dataset, dataset_metrics in sorted(judge_metrics.items()):
            for system, system_metrics in sorted(dataset_metrics.items()):
                for axis, axis_metrics in sorted(system_metrics.items()):
                    for rating, values in axis_metrics.items():
                        edit_quality.append(
                            {
                                "judge_source": source,
                                "judge": judge,
                                "dataset": dataset,
                                "system": system,
                                "axis": axis,
                                "rating": rating,
                                **values,
                            }
                        )

    sentence_adequacy = []
    for judge, judge_metrics in sorted(metrics["sentence_adequacy"].items()):
        for dataset, dataset_metrics in sorted(judge_metrics.items()):
            for system, values in sorted(dataset_metrics.items()):
                sentence_adequacy.append(
                    {
                        "judge_source": source,
                        "judge": judge,
                        "dataset": dataset,
                        "system": system,
                        **values,
                    }
                )

    judge_agreement = []
    for dataset, dataset_metrics in sorted(metrics["judge_agreement"].items()):
        judge_agreement.append(
            {
                "judge_source": source,
                "dataset": dataset,
                "system": "ALL",
                **dataset_metrics["overall"],
            }
        )
        for system, values in sorted(dataset_metrics["systems"].items()):
            judge_agreement.append(
                {
                    "judge_source": source,
                    "dataset": dataset,
                    "system": system,
                    **values,
                }
            )

    paired_label_good = []
    for judge, judge_metrics in sorted(metrics["paired_label_good"].items()):
        for dataset, dataset_metrics in sorted(judge_metrics.items()):
            for comparison, values in sorted(dataset_metrics["comparisons"].items()):
                system_a, system_b = comparison.split("_minus_", 1)
                paired_label_good.append(
                    {
                        "judge_source": source,
                        "judge": judge,
                        "dataset": dataset,
                        "system_a": system_a,
                        "system_b": system_b,
                        **values,
                    }
                )

    paired_sentence_adequacy = []
    for judge, judge_metrics in sorted(metrics["paired_sentence_adequacy"].items()):
        for dataset, dataset_metrics in sorted(judge_metrics.items()):
            for comparison, values in sorted(dataset_metrics["comparisons"].items()):
                system_a, system_b = comparison.split("_minus_", 1)
                paired_sentence_adequacy.append(
                    {
                        "judge_source": source,
                        "judge": judge,
                        "dataset": dataset,
                        "system_a": system_a,
                        "system_b": system_b,
                        **values,
                    }
                )

    compact_summary = []
    datasets = sorted(metrics["judge_agreement"])
    for dataset in datasets:
        systems = sorted(metrics["judge_agreement"][dataset]["systems"])
        for system in systems:
            primary_quality = metrics["acceptability"]["primary"][dataset][system]
            second_quality = metrics["acceptability"]["second"][dataset][system]
            primary_adequacy = metrics["sentence_adequacy"]["primary"][dataset][system]
            second_adequacy = metrics["sentence_adequacy"]["second"][dataset][system]
            agreement = metrics["judge_agreement"][dataset]["systems"][system]
            compact_summary.append(
                {
                    "judge_source": source,
                    "dataset": dataset,
                    "system": system,
                    "n_edits": agreement["n_edits"],
                    "n_items": primary_adequacy["n_items"],
                    "span_agreement_pct": agreement["span_agreement_pct"],
                    "label_agreement_pct": agreement["label_agreement_pct"],
                    "primary_span_good_pct": primary_quality["span"]["Good"]["pct"],
                    "primary_label_good_pct": primary_quality["label"]["Good"]["pct"],
                    "primary_adequacy_pct": primary_adequacy["adequacy_pct"],
                    "second_span_good_pct": second_quality["span"]["Good"]["pct"],
                    "second_label_good_pct": second_quality["label"]["Good"]["pct"],
                    "second_adequacy_pct": second_adequacy["adequacy_pct"],
                }
            )

    return {
        "compact_summary": compact_summary,
        "edit_quality": edit_quality,
        "sentence_adequacy": sentence_adequacy,
        "judge_agreement": judge_agreement,
        "paired_label_good": paired_label_good,
        "paired_sentence_adequacy": paired_sentence_adequacy,
    }


def compute_paper_tables(
    corpus: JudgeVerdictCorpus,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute metrics and tables in one path rooted in complete raw verdict rows."""

    metrics = compute_judge_metrics(corpus, samples=samples, seed=seed)
    return {
        "schema_version": "metc-raw-judge-tables-v1",
        "judge_source": corpus.judge_source,
        "metrics": metrics,
        "tables": paper_table_rows(metrics),
    }
