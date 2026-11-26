from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from metc_artifact.systems import (
    SystemNotReadyError,
    SystemRunner,
    UnsupportedSystemExecutionError,
)


def test_preflight_reports_every_missing_runtime_requirement(tmp_path: Path) -> None:
    runner = SystemRunner(artifact_root=tmp_path, environment={"PATH": os.environ["PATH"]})
    with pytest.raises(SystemNotReadyError) as caught:
        runner.prepare("jp_errant", "zh")

    issues = {(issue.category, issue.name) for issue in caught.value.issues}
    assert ("environment", "JP_ERRANT_PARALLEL") in issues
    assert ("asset", "stanza_resources") in issues
    assert ("asset", "zh_gsdltp_model") in issues
    assert ("value", "source") in issues
    assert ("value", "correction") in issues
    assert ("value", "output") in issues


def test_inapplicable_system_is_rejected_before_launch(tmp_path: Path) -> None:
    runner = SystemRunner(artifact_root=tmp_path, environment={})
    with pytest.raises(ValueError, match="does not apply"):
        runner.prepare("errant", "de")


def test_dataset_specific_assets_are_required_only_where_used(tmp_path: Path) -> None:
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    source = tmp_path / "source.txt"
    correction = tmp_path / "correction.txt"
    source.write_text("She go.\n", encoding="utf-8")
    correction.write_text("She goes.\n", encoding="utf-8")
    runner = SystemRunner(
        artifact_root=tmp_path,
        environment={
            "JP_ERRANT_PARALLEL": sys.executable,
            "JP_ERRANT_STANZA_RESOURCES": str(stanza),
            "PATH": os.environ["PATH"],
        },
    )

    prepared = runner.prepare(
        "jp_errant",
        "en",
        values={
            "source": source,
            "correction": correction,
            "output": tmp_path / "output.m2",
        },
    )
    assert "-lang" in prepared.argv
    assert prepared.argv[-1] == "en"


def test_runner_refuses_to_launch_historical_registry_commands(tmp_path: Path) -> None:
    runner = SystemRunner(artifact_root=tmp_path, environment={})

    with pytest.raises(UnsupportedSystemExecutionError, match="canonical E2E execution"):
        runner.run("errant", "en_conll14_ann0")
