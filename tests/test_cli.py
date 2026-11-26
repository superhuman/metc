import json
import subprocess
import sys
from dataclasses import dataclass

import pytest

from metc import cli
from metc.cli import _iter_jsonl, build_parser


@dataclass
class FakeResult:
    source: str
    tokenized: bool
    cleanup_semantic: bool

    def to_dict(self):
        return {
            "source": self.source,
            "tokenized": self.tokenized,
            "cleanup_semantic": self.cleanup_semantic,
        }


class FakePipeline:
    def __init__(self):
        self.calls = []

    def classify_batch(self, pairs, *, tokenized, cleanup_semantic):
        self.calls.append((list(pairs), tokenized, cleanup_semantic))
        return [FakeResult(source, tokenized, cleanup_semantic) for source, _target in pairs]


def test_cli_defaults_to_released_inference_limit():
    args = build_parser().parse_args(
        [
            "classify",
            "--model",
            "/tmp/model",
            "--source",
            "a",
            "--target",
            "b",
        ]
    )
    assert args.max_sequence_length == 512
    assert args.device == "auto"
    assert args.batch_size == 8
    assert args.inference_profile == "portable"


def test_cli_exposes_fail_closed_paper_profile():
    args = build_parser().parse_args(
        [
            "label",
            "--model",
            "/tmp/model",
            "--inference-profile",
            "paper",
            "--annotated",
            "[EDIT]a=>b[/EDIT]",
        ]
    )
    assert args.inference_profile == "paper"


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_module_entrypoint_is_executable():
    completed = subprocess.run(
        [sys.executable, "-m", "metc.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert "classify" in completed.stdout


def test_jsonl_flags_must_be_actual_booleans(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"source": "a", "target": "b", "tokenized": "false"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        list(_iter_jsonl(str(path)))


def test_jsonl_streams_multiple_batches_preserving_order_and_row_flags(tmp_path, monkeypatch):
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    rows = [
        {"source": "zero", "target": "ZERO"},
        {"source": "one", "target": "ONE", "tokenized": True},
        {"source": "two", "target": "TWO", "cleanup_semantic": True},
        {
            "source": "three",
            "target": "THREE",
            "tokenized": False,
            "cleanup_semantic": False,
        },
        {"source": "four", "target": "FOUR", "tokenized": True},
        {"source": "five", "target": "FIVE"},
        {"source": "six", "target": "SIX", "cleanup_semantic": False},
    ]
    input_path.write_text(
        "\n".join(
            [
                json.dumps(rows[0]),
                "",
                json.dumps(rows[1]),
                json.dumps(rows[2]),
                "   ",
                *(json.dumps(row) for row in rows[3:]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "_make_pipeline", lambda _args: pipeline)

    assert (
        cli.main(
            [
                "classify",
                "--model",
                "/tmp/model",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--batch-size",
                "2",
                "--cleanup-semantic",
            ]
        )
        == 0
    )

    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in outputs] == [row["source"] for row in rows]
    assert [(row["tokenized"], row["cleanup_semantic"]) for row in outputs] == [
        (False, True),
        (True, True),
        (False, True),
        (False, False),
        (True, True),
        (False, True),
        (False, False),
    ]
    assert sum(len(pairs) for pairs, _tokenized, _cleanup in pipeline.calls) == len(rows)
    assert all(len(pairs) <= 2 for pairs, _tokenized, _cleanup in pipeline.calls)


def test_jsonl_errors_report_physical_line_numbers(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '\n{"source": "valid", "target": "valid"}\n\n{"source": 1, "target": "x"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 4 requires string source and target fields"):
        list(_iter_jsonl(str(path)))


@pytest.mark.parametrize("alias_kind", ["same", "lexical", "symlink"])
def test_cli_rejects_same_input_and_output_without_truncating(alias_kind, tmp_path, monkeypatch):
    input_path = tmp_path / "pairs.jsonl"
    original = json.dumps({"source": "a", "target": "b"}) + "\n"
    input_path.write_text(original, encoding="utf-8")

    if alias_kind == "same":
        output_path = input_path
    elif alias_kind == "lexical":
        nested = tmp_path / "nested"
        nested.mkdir()
        output_path = nested / ".." / input_path.name
    else:
        output_path = tmp_path / "pairs-alias.jsonl"
        output_path.symlink_to(input_path)

    pipeline_constructed = False

    def fail_if_constructed(_args):
        nonlocal pipeline_constructed
        pipeline_constructed = True
        raise AssertionError("pipeline must not be constructed for an unsafe output path")

    monkeypatch.setattr(cli, "_make_pipeline", fail_if_constructed)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "classify",
                "--model",
                "/tmp/model",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )

    assert not pipeline_constructed
    assert input_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_cli_rejects_invalid_batch_size(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "classify",
                "--model",
                "/tmp/model",
                "--input",
                "pairs.jsonl",
                "--batch-size",
                value,
            ]
        )
