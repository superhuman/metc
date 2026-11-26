import csv
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def test_sanitized_training_recipe_is_the_token_classification_run():
    recipe = json.loads((ROOT / "training" / "recipe.json").read_text(encoding="utf-8"))

    assert recipe["schema_version"] == "metc-training-recipe-v1"
    assert recipe["model"] == {
        "architecture": "ModernBertForTokenClassification",
        "base_encoder": "jhu-clsp/mmBERT-small",
        "edit_close_token": "[/EDIT]",
        "edit_open_token": "[EDIT]",
        "max_length": 256,
        "num_labels": 42,
    }
    assert "source_recipe" not in recipe
    assert len(recipe["data"]["training_sources"]) == 9
    assert recipe["selection"]["metric"] == "f1_macro"
    assert recipe["run_summary"] == {
        "examples": {
            "total": 1_699_176,
            "training": 1_529_258,
            "validation": 169_918,
        },
        "reported_tier_rows_rounded_to_nearest_1000": {
            "english": 438_000,
            "five_higher_resource_languages": 450_000,
            "long_tail": 811_000,
        },
        "selected_checkpoint": {
            "step": 23_000,
            "validation_f1_macro": 0.7583,
        },
        "training_stop": {
            "epoch": 2.176,
            "reason": "early_stopping_patience_reached",
            "step": 26_000,
        },
    }
    assert (
        recipe["run_summary"]["examples"]["training"]
        + recipe["run_summary"]["examples"]["validation"]
        == recipe["run_summary"]["examples"]["total"]
    )
    assert recipe["source_test_files_used_for_checkpoint_selection"] is False


def test_sanitized_training_recipe_contains_no_private_path():
    recipe = json.loads((ROOT / "training" / "recipe.json").read_text(encoding="utf-8"))
    strings = tuple(_all_strings(recipe))
    assert not any(value.startswith("/") for value in strings)
    assert not any("volumes/" in value.lower() for value in strings)


def test_training_corpus_summary_matches_reported_aggregate_values():
    path = ROOT / "training" / "training_corpus_summary.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)

    assert rows == [
        ["metric", "overall", "english", "five_languages", "long_tail", "unit"],
        ["Spelling", "30.4", "10.0", "32.5", "37.5", "percent_of_edits"],
        ["Punctuation", "24.0", "24.1", "23.6", "24.2", "percent_of_edits"],
        ["Grammar", "22.5", "40.5", "23.8", "15.3", "percent_of_edits"],
        ["Formatting", "10.8", "6.2", "12.1", "12.2", "percent_of_edits"],
        ["Other", "8.5", "7.2", "6.6", "9.7", "percent_of_edits"],
        ["Rephrasing", "2.8", "10.6", "0.4", "0.6", "percent_of_edits"],
        ["SentenceStructure", "0.9", "1.5", "1.0", "0.7", "percent_of_edits"],
        ["EditsPerSentence", "2.69", "2.28", "2.07", "3.25", "mean"],
    ]


def test_wheel_requires_authored_and_third_party_license_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "UNICODE_LICENSE.txt",
    ]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    notice_text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright 2026 Superhuman Platform Inc." in notice_text


def test_runtime_does_not_install_development_group_by_default():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["tool"]["uv"]["default-groups"] == []
    assert set(configuration["dependency-groups"]["dev"]) == {
        "pytest==8.4.2",
        "ruff==0.14.10",
    }
