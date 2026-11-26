from __future__ import annotations

from metc_artifact.judge_jobs import build_judge_jobs
from metc_artifact.schema import dataset_spec


def test_judge_job_builder_uses_stable_physical_edit_ids() -> None:
    spec = dataset_spec("en")
    row = {
        "sent_idx": 4,
        "src": "a",
        "tgt": "b",
        **{system: [] for system in spec.systems},
    }
    row["gold"] = [[0, 1, "mapped", "native", "b"]]

    build = build_judge_jobs([row], spec)

    assert [job["edit_id"] for job in build.jobs] == ["metc-edit-v1:en:4:gold:0"]
    assert build.jobs[0]["edit_diff"] == "'a' => 'b'"


def test_judge_job_builder_applies_paper_prefilters() -> None:
    spec = dataset_spec("en")
    row = {
        "idx": 0,
        "src": " a",
        "tgt": " a",
        **{system: [] for system in spec.systems},
    }
    row["gold"] = [
        [0, 1, "mapped", "space", ""],
        [1, 2, "mapped", "noop", "a"],
    ]

    build = build_judge_jobs([row], spec)

    assert build.jobs == ()
    assert build.whitespace_only["gold"] == 0
    assert build.nonjudgeable["gold"] == 2
