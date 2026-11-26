"""Table-ready records derived from raw human-evaluation metrics."""

from __future__ import annotations

from typing import Any, Mapping

from .human import HumanStudyCorpus, compute_human_metrics


def _source(metrics: Mapping[str, Any]) -> str:
    source = metrics.get("human_source")
    if source != "frozen_authoritative":
        raise ValueError("human metrics lack frozen_authoritative provenance")
    return str(source)


def human_table_rows(metrics: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten a raw human metrics document into deterministic table rows."""

    source = _source(metrics)
    dataset = str(metrics["dataset"])
    systems = list(metrics["human_edit_quality"])

    compact_summary = []
    edit_quality = []
    sentence_adequacy = []
    for system in systems:
        quality = metrics["human_edit_quality"][system]
        adequacy = metrics["sentence_adequacy"][system]
        restricted = metrics["judge_label_good_on_human_edits"][system]
        compact_summary.append(
            {
                "human_source": source,
                "dataset": dataset,
                "system": system,
                "span_good_pct": quality["span"]["Good"]["pct"],
                "label_good_pct": quality["label"]["Good"]["pct"],
                "label_acceptable_pct": quality["label"]["Acceptable"]["pct"],
                "label_bad_pct": quality["label"]["Bad"]["pct"],
                "adequacy_pct": adequacy["adequacy_pct"],
                "adequacy_ci_low": adequacy["ci_low"],
                "adequacy_ci_high": adequacy["ci_high"],
                "primary_label_good_pct": restricted["primary"]["label_good_pct"],
                "second_label_good_pct": restricted["second"]["label_good_pct"],
            }
        )
        for axis in ("span", "label"):
            for rating, values in quality[axis].items():
                edit_quality.append(
                    {
                        "human_source": source,
                        "dataset": dataset,
                        "system": system,
                        "axis": axis,
                        "rating": rating,
                        **values,
                    }
                )
        sentence_adequacy.append(
            {
                "human_source": source,
                "dataset": dataset,
                "system": system,
                **adequacy,
            }
        )

    agreement = []
    inter = metrics["inter_annotator_agreement"]
    agreement.append(
        {
            "human_source": source,
            "dataset": dataset,
            "comparison": "human_vs_human",
            "system": "ALL",
            "n_verdicts": inter["n_double_scorable_edits"],
            "span_agreement_pct": inter["span_agreement_pct"],
            "label_agreement_pct": inter["label_agreement_pct"],
        }
    )
    for system, values in inter["systems"].items():
        agreement.append(
            {
                "human_source": source,
                "dataset": dataset,
                "comparison": "human_vs_human",
                "system": system,
                **values,
            }
        )
    for judge, judge_metrics in metrics["human_vs_judges"].items():
        agreement.append(
            {
                "human_source": source,
                "dataset": dataset,
                "comparison": f"human_vs_{judge}",
                "system": "ALL",
                "n_verdicts": judge_metrics["n_human_verdicts"],
                "span_agreement_pct": judge_metrics["span_agreement_pct"],
                "label_agreement_pct": judge_metrics["label_agreement_pct"],
            }
        )
        for system, values in judge_metrics["systems"].items():
            agreement.append(
                {
                    "human_source": source,
                    "dataset": dataset,
                    "comparison": f"human_vs_{judge}",
                    "system": system,
                    **values,
                }
            )
    cross = metrics["judge_vs_judge_on_human_edits"]
    agreement.append(
        {
            "human_source": source,
            "dataset": dataset,
            "comparison": "primary_vs_second",
            "system": "ALL",
            "n_verdicts": cross["n_unique_edits"],
            "span_agreement_pct": cross["span_agreement_pct"],
            "label_agreement_pct": cross["label_agreement_pct"],
        }
    )

    comparisons = []
    for metric, metric_values in metrics["system_comparisons"].items():
        for sensitivity, sensitivity_values in metric_values.items():
            for comparison, values in sensitivity_values.items():
                comparisons.append(
                    {
                        "human_source": source,
                        "dataset": dataset,
                        "metric": metric,
                        "analysis": sensitivity,
                        "comparison": comparison,
                        **values,
                    }
                )

    sensitivity = []
    for analysis, analysis_values in metrics["adequacy_sensitivity"].items():
        for system, values in analysis_values.items():
            sensitivity.append(
                {
                    "human_source": source,
                    "dataset": dataset,
                    "analysis": analysis,
                    "system": system,
                    **values,
                }
            )

    return {
        "compact_summary": compact_summary,
        "edit_quality": edit_quality,
        "sentence_adequacy": sentence_adequacy,
        "agreement": agreement,
        "system_comparisons": comparisons,
        "adequacy_sensitivity": sensitivity,
    }


def compute_human_tables(
    corpus: HumanStudyCorpus,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute raw human metrics and table records in one audited path."""

    metrics = compute_human_metrics(corpus, samples=samples, seed=seed)
    return {
        "schema_version": "metc-raw-human-tables-v1",
        "human_source": metrics["human_source"],
        "dataset": corpus.dataset_id,
        "metrics": metrics,
        "tables": human_table_rows(metrics),
    }
