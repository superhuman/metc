from __future__ import annotations

import sys
from pathlib import Path

import pytest

from metc_artifact.adapters import AdapterInputError, PairRecord
from metc_artifact.adapters.metc import run_metc
from metc_artifact.schema import dataset_spec


def _fake_metc_package(root: Path) -> Path:
    package = root / "metc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd', required=True)\n"
        "c=s.add_parser('classify')\n"
        "c.add_argument('--model'); c.add_argument('--input'); c.add_argument('--output')\n"
        "c.add_argument('--batch-size'); c.add_argument('--device')\n"
        "c.add_argument('--inference-profile')\n"
        "a=p.parse_args()\n"
        "rows=[json.loads(x) for x in open(a.input, encoding='utf-8') if x.strip()]\n"
        "with open(a.output, 'w', encoding='utf-8') as out:\n"
        "  for r in rows:\n"
        "    if r['tokenized']:\n"
        "      ann={'startToken':1,'endToken':2,'label':'Grammar','sublabel':'Verb',"
        "'replacement':'gehe'}\n"
        "    else:\n"
        "      ann={'start':3,'end':5,'label':'Grammar','sublabel':'Verb',"
        "'replacement':'goes'}\n"
        "    out.write(json.dumps({'annotations':[ann]})+'\\n')\n",
        encoding="utf-8",
    )
    return root


def test_metc_adapter_runs_batch_and_converts_utf16_to_codepoints(tmp_path: Path) -> None:
    fake = _fake_metc_package(tmp_path / "fake")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = dataset_spec("en")
    pair = PairRecord(
        dataset_id=spec.dataset_id,
        item_id="1",
        source="😂 go",
        correction="😂 goes",
    )
    records = run_metc(
        [pair],
        spec,
        python=sys.executable,
        checkpoint=checkpoint,
        environment={"PYTHONPATH": str(fake)},
    )
    edit = records[0].edits[0]
    assert (edit.start, edit.end) == (2, 4)
    assert edit.native_label == "Grammar/Verb"
    assert edit.mapped_label == "Wci"


def test_metc_adapter_uses_token_mode_for_ukraine(tmp_path: Path) -> None:
    fake = _fake_metc_package(tmp_path / "fake")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = dataset_spec("uk")
    pair = PairRecord(
        dataset_id=spec.dataset_id,
        item_id="0",
        source="Я йти",
        correction="Я йду",
        source_tokens=("Я", "йти"),
        correction_tokens=("Я", "йду"),
    )
    records = run_metc(
        [pair],
        spec,
        python=sys.executable,
        checkpoint=checkpoint,
        environment={"PYTHONPATH": str(fake)},
    )
    assert (records[0].edits[0].start, records[0].edits[0].end) == (1, 2)
    assert records[0].edits[0].mapped_label == "G/Tense"


def test_metc_char_mode_splices_subtoken_edit_into_full_token(tmp_path: Path) -> None:
    fake = _fake_metc_package(tmp_path / "fake")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = dataset_spec("de")
    pair = PairRecord(
        dataset_id=spec.dataset_id,
        item_id="0",
        source="Er go.",
        correction="Er goes.",
        source_tokens=("Er", "go."),
        correction_tokens=("Er", "goes."),
    )
    records = run_metc(
        [pair],
        spec,
        python=sys.executable,
        checkpoint=checkpoint,
        environment={"PYTHONPATH": str(fake)},
    )
    edit = records[0].edits[0]
    assert (edit.start, edit.end) == (1, 2)
    assert edit.replacement == "goes."
    assert edit.char_replacement == "goes"


def test_portable_metc_requires_explicit_nonpaper_override(tmp_path: Path) -> None:
    spec = dataset_spec("en")
    with pytest.raises(AdapterInputError, match="known not to reproduce"):
        run_metc(
            [],
            spec,
            python=sys.executable,
            checkpoint=tmp_path,
            inference_profile="portable",
        )
