from __future__ import annotations

import sys
from pathlib import Path

from metc_artifact.adapters import PairRecord
from metc_artifact.adapters.errant import run_errant
from metc_artifact.schema import dataset_spec


def test_errant_adapter_uses_external_worker_and_normalizes_labels(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "rows=[json.loads(x) for x in open(a.input, encoding='utf-8')]\n"
        "with open(a.output, 'w', encoding='utf-8') as out:\n"
        "  for r in rows:\n"
        "    out.write(json.dumps({'item_id':r['item_id'],'edits':[{'start':4,'end':6,"
        "'replacement':'goes','native_label':'R:VERB:SVA'}]})+'\\n')\n",
        encoding="utf-8",
    )
    spec = dataset_spec("en")
    pair = PairRecord(
        dataset_id=spec.dataset_id,
        item_id="4",
        source="She go.",
        correction="She goes.",
    )
    record = run_errant([pair], spec, python=sys.executable, worker=worker)[0]
    assert record.system == "errant"
    assert record.edits[0].mapped_label == "SVA"
    assert (record.edits[0].start, record.edits[0].end) == (4, 6)


def test_default_worker_imports_external_errant_not_sibling_module(tmp_path: Path) -> None:
    package = tmp_path / "errant"
    package.mkdir()
    (package / "__init__.py").write_text(
        "class Token:\n"
        "  def __init__(self,text,idx): self.text=text; self.idx=idx\n"
        "class Doc(list): pass\n"
        "class Edit:\n"
        "  type='R:VERB:SVA'; o_start=1; o_end=2; c_str='goes'\n"
        "class Annotator:\n"
        "  def parse(self,text):\n"
        "    words=text.split(); out=Doc(); pos=0\n"
        "    for word in words:\n"
        "      pos=text.index(word,pos); out.append(Token(word,pos)); pos+=len(word)\n"
        "    return out\n"
        "  def annotate(self,original,correction): return [Edit()]\n"
        "def load(language): return Annotator()\n",
        encoding="utf-8",
    )
    spec = dataset_spec("en")
    pair = PairRecord(spec.dataset_id, "0", "She go", "She goes")
    record = run_errant(
        [pair],
        spec,
        python=sys.executable,
        environment={"PYTHONPATH": str(tmp_path)},
    )[0]
    assert record.edits[0].mapped_label == "SVA"
    assert (record.edits[0].start, record.edits[0].end) == (4, 6)
