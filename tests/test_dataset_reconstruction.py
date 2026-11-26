from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path

import pytest

from metc_artifact.datasets.m2 import reconstruct_conll, reconstruct_kolla
from metc_artifact.datasets.merlin import reconstruct_merlin
from metc_artifact.datasets.mucgec import (
    PARALLEL_TO_M2_SHA256,
    UPSTREAM_TOKENIZER_SHA256,
    cherrant_category,
    generate_cherrant_reference,
)
from metc_artifact.datasets.registry import assert_semantically_equal
from metc_artifact.datasets.uagec import reconstruct_uagec


def test_conll_reference_uses_requested_annotator_and_character_offsets(tmp_path: Path) -> None:
    m2 = tmp_path / "combined.m2"
    m2.write_text(
        "S I has apples .\n"
        "A 1 2|||Vt|||have|||REQUIRED|||-NONE-|||0\n"
        "A 2 3|||Nn|||an apple|||REQUIRED|||-NONE-|||1\n\n"
        "S Already correct .\n"
        "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n",
        encoding="utf-8",
    )
    ann0 = reconstruct_conll(m2, 0)
    ann1 = reconstruct_conll(m2, 1)
    assert ann0 == [
        {
            "sent_idx": 0,
            "src": "I has apples .",
            "tgt": "I have apples .",
            "gold": [[2, 5, "Vt", "Vt", "have"]],
        }
    ]
    assert ann1[0]["tgt"] == "I has an apple ."
    assert ann1[0]["gold"] == [[6, 12, "Nn", "Nn", "an apple"]]


def test_kolla_same_position_edits_have_deterministic_application_order(tmp_path: Path) -> None:
    m2 = tmp_path / "KoLLA_multi-refs.m2"
    m2.write_text(
        "S 시작\n"
        "A 1 1|||M:X|||하나|||REQUIRED|||-NONE-|||0\n"
        "A 1 1|||M:Y|||둘|||REQUIRED|||-NONE-|||0\n",
        encoding="utf-8",
    )
    rows = reconstruct_kolla(m2)
    assert rows[0]["cor_tokens"] == ["시작", "하나", "둘"]
    assert [edit["raw_label"] for edit in rows[0]["gold"]] == ["M:X", "M:Y"]


def _merlin_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<basic-transcription>
  <basic-body>
    <common-timeline><tli id="T0"/><tli id="T1"/><tli id="T2"/></common-timeline>
    <tier category="tok"><event start="T0" end="T1">same</event><event start="T1" end="T2">bad</event></tier>
    <tier category="TH1"><event start="T0" end="T1">same</event><event start="T1" end="T2">good</event></tier>
    <tier category="sentence"><event start="T0" end="T2">sentence</event></tier>
    <tier category="markable_scheme"><event start="T0" end="T2">G_Test</event></tier>
    <tier category="G_Test_kind"><event start="T0" end="T2">changed</event></tier>
  </basic-body>
</basic-transcription>
"""


def test_merlin_minimizes_shared_edge_tokens(tmp_path: Path) -> None:
    czech = tmp_path / "bundle" / "czech"
    czech.mkdir(parents=True)
    (czech / "essay.exb").write_text(_merlin_xml(), encoding="utf-8")
    rows = reconstruct_merlin(tmp_path, "czech")
    assert rows == [
        {
            "idx": 0,
            "essay_id": "essay",
            "orig_tokens": ["same", "bad"],
            "th1_tokens": ["same", "good"],
            "gold": [
                {
                    "orig_start": 1,
                    "orig_end": 2,
                    "th1_start": 1,
                    "th1_end": 2,
                    "raw_label": "G_Test/changed",
                    "target": "G_Test",
                }
            ],
        }
    ]


def test_german_merlin_uses_unique_sorted_public_boyd_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    german = tmp_path / "merlin" / "german"
    german.mkdir(parents=True)
    (german / "b.exb").write_text(_merlin_xml(), encoding="utf-8")
    (german / "a.exb").write_text(_merlin_xml(), encoding="utf-8")
    boyd = tmp_path / "boyd"
    selection = boyd / "data" / "source" / "merlin-id-ctok-zh1.txt.dev"
    selection.parent.mkdir(parents=True)
    selection.write_text("b one\na two\nb three\n", encoding="utf-8")
    monkeypatch.setattr(
        "metc_artifact.datasets.merlin.BOYD_DEVELOPMENT_IDS_SHA256",
        hashlib.sha256(selection.read_bytes()).hexdigest(),
    )
    rows = reconstruct_merlin(tmp_path / "merlin", "german", boyd_root=boyd)
    assert len(rows) == 2
    assert [row["idx"] for row in rows] == [0, 1]


def test_uagec_reconstructs_token_span_replacement_from_inline_character_edit(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "metadata.csv").write_text(
        "id,author_id,is_native,region,gender,occupation,submission_type,"
        "source_language,annotator_id,partition,is_sensitive\n"
        "0001,a,1,r,g,o,essay,,1,test,0\n",
        encoding="utf-8",
    )
    annotated = tmp_path / "data" / "gec-only" / "test" / "annotated"
    annotated.mkdir(parents=True)
    (annotated / "0001.a1.ann").write_text(
        "Це {тес=>текс:::error_type=Spelling}т.\n", encoding="utf-8"
    )
    rows = reconstruct_uagec(tmp_path)
    assert rows[0]["orig_tokens"] == ["Це", "тест."]
    assert rows[0]["cor_tokens"] == ["Це", "текст."]
    assert rows[0]["gold"][0]["replacement"] == "текст."
    assert rows[0]["gold"][0]["orig_start"] == 1
    assert rows[0]["gold"][0]["orig_end"] == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("R:VERB", "VERB"), ("W", "W"), ("S:UNKNOWN", "OTHER"), ("", "")],
)
def test_cherrant_category(raw: str, expected: str) -> None:
    assert cherrant_category(raw) == expected


def test_failed_cherrant_generation_does_not_clobber_cached_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "checkout" / "scorers" / "ChERRANT"
    modules = source_root / "modules"
    modules.mkdir(parents=True)
    tokenizer = modules / "tokenizer.py"
    tokenizer.write_text("upstream tokenizer", encoding="utf-8")
    entrypoint = source_root / "parallel_to_m2.py"
    entrypoint.write_text("upstream entrypoint", encoding="utf-8")
    work_dir = tmp_path / "cache" / "generated" / "zh_mucgec"
    work_dir.mkdir(parents=True)
    cached = work_dir / "cherrant_ref_zh.m2"
    cached.write_bytes(b"previous verified reference")

    monkeypatch.setattr(
        "metc_artifact.datasets.mucgec._one",
        lambda _root, _pattern, _description: source_root,
    )

    def fake_sha256(path: Path) -> str:
        return {
            tokenizer: UPSTREAM_TOKENIZER_SHA256,
            entrypoint: PARALLEL_TO_M2_SHA256,
        }[path]

    monkeypatch.setattr("metc_artifact.datasets.mucgec._sha256", fake_sha256)

    def fake_parallel(_root: Path, output: Path) -> Path:
        output.write_text("parallel rows\n", encoding="utf-8")
        return output

    monkeypatch.setattr(
        "metc_artifact.datasets.mucgec.write_first_reference_parallel",
        fake_parallel,
    )

    def failed_run(command, **_kwargs):
        candidate = Path(command[command.index("-o") + 1])
        candidate.write_bytes(b"partial output")
        return SimpleNamespace(returncode=1, stderr="synthetic failure")

    monkeypatch.setattr("metc_artifact.datasets.mucgec.subprocess.run", failed_run)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        generate_cherrant_reference(tmp_path, work_dir, Path("python"))

    assert cached.read_bytes() == b"previous verified reference"
    assert not list(work_dir.glob(".cherrant-run-*"))


def test_semantic_comparison_ignores_batch_order_and_uk_diagnostics() -> None:
    base = {
        "idx": 0,
        "doc_id": "0001",
        "para_idx": 0,
        "orig_tokens": ["a"],
        "cor_tokens": ["b"],
        "gold": [
            {
                "orig_start": 0,
                "orig_end": 1,
                "raw_label": "Spelling",
                "target": "Spelling",
                "replacement": "b",
                "char_start": 0,
                "char_end": 1,
            }
        ],
    }
    frozen = {**base, "gold": [{**base["gold"][0], "char_start": 99}]}
    assert_semantically_equal("uk_uagec", [base], [frozen])
