from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from metc_artifact.adapters import AdapterOutputError, AdapterUnavailableError, PairRecord
from metc_artifact.adapters.cherrant import (
    EXPECTED_COMMIT as CHERRANT_COMMIT,
    EXPECTED_LTP_REVISION,
    EXPECTED_PATCHES as CHERRANT_PATCHES,
    run_cherrant_reference,
)
from metc_artifact.adapters.gerrant import (
    apply_gerrant_paper_compat,
    main as gerrant_main,
    run_gerrant,
)
from metc_artifact.adapters.jp_errant import (
    EXPECTED_COMMIT as JP_COMMIT,
    EXPECTED_TRANSFORMATIONS as JP_PATCHES,
    main as jp_main,
    run_jp_errant,
    run_jp_errant_bulk,
    run_jp_errant_sharded_views,
    run_jp_errant_views,
)
from metc_artifact.adapters.m2 import (
    M2Edit,
    M2Sentence,
    align_m2_by_position,
    normalized_m2_edits,
)
from metc_artifact.schema import dataset_spec


def _write_patch_manifest(path: Path, *, commit: str, patches: dict, **extra: str) -> Path:
    modifications_key = "transformations" if commit == JP_COMMIT else "patches"
    path.write_text(
        json.dumps({"upstream_commit": commit, modifications_key: patches, **extra}),
        encoding="utf-8",
    )
    return path


def test_gerrant_reproduces_historical_first_duplicate_source_materialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gerrant"
    root.mkdir()
    script = root / "gerrant_de.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); a=p.parse_args()\n"
        "src=open(a.orig, encoding='utf-8').read().splitlines()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  for i,s in enumerate(src):\n"
        "    out.write('S '+s+'\\n')\n"
        "    if i in (1,2): out.write('A 1 2|||R:SPELL|||geht|||REQUIRED|||-NONE-|||0\\n')\n"
        "    else: out.write('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\\n')\n"
        "    out.write('\\n')\n",
        encoding="utf-8",
    )
    spec = dataset_spec("de")
    pairs = [
        PairRecord(spec.dataset_id, "0", "Er geht", "Er geht", ("Er", "geht"), ("Er", "geht")),
        PairRecord(spec.dataset_id, "1", "Er geht", "Er geht.", ("Er", "geht"), ("Er", "geht.")),
        PairRecord(spec.dataset_id, "2", "Sie geht", "Sie geht", ("Sie", "geht"), ("Sie", "geht")),
    ]
    raw_records = run_gerrant(
        pairs,
        spec,
        python=sys.executable,
        gerrant_root=root,
        validate_provenance=False,
    )
    assert [len(record.edits) for record in raw_records] == [0, 1, 1]
    records = apply_gerrant_paper_compat(pairs, raw_records)
    assert [len(record.edits) for record in records] == [0, 0, 1]
    assert records[2].edits[0].mapped_label == "O_Graph"


def _jp_script(path: Path) -> Path:
    path.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "src=open(a.orig,encoding='utf-8').read().splitlines()\n"
        "labels={'en':'R:VERB','de':'R:ADP -> ADP','cs':'R:ADP -> ADP',"
        "'ko':'R:NOUN -> NOUN','uk':'R:ADP -> ADP','zh':'R:NOUN -> NOUN'}\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  for s in src:\n"
        "    toks=' '.join(s) if a.lang=='zh' else s\n"
        "    out.write('S '+toks+'\\n')\n"
        "    out.write('A 0 1|||'+labels[a.lang]+'|||X|||REQUIRED|||-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("dataset", "source", "correction", "tokens", "mapped"),
    [
        ("en", "go now", "goes now", None, "Wci"),
        ("en1", "go now", "goes now", None, "Wci"),
        ("de", "in Haus", "im Haus", ("in", "Haus"), "G_Prep"),
        ("cs", "v dům", "v domě", ("v", "dům"), "G_Prep"),
        ("ko", "나 집", "나는 집", ("나", "집"), "R:NOUN -> NOUN"),
        ("uk", "у дім", "в дім", ("у", "дім"), "G/Prep"),
        ("zh", "你好", "您好", None, "NOUN"),
    ],
)
def test_jp_errant_adapter_covers_every_paper_setting(
    tmp_path: Path,
    dataset: str,
    source: str,
    correction: str,
    tokens: tuple[str, ...] | None,
    mapped: str,
) -> None:
    spec = dataset_spec(dataset)
    script = _jp_script(tmp_path / f"jp-{dataset}.py")
    patch_manifest = _write_patch_manifest(
        tmp_path / f"jp-{dataset}.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir(exist_ok=True)
    gsdltp = tmp_path / "gsdltp"
    gsdltp.mkdir(exist_ok=True)
    pair = PairRecord(
        spec.dataset_id,
        "0",
        source,
        correction,
        tokens,
        tuple(correction.split(" ")) if tokens is not None else None,
    )
    record = run_jp_errant(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=patch_manifest,
        stanza_resources=stanza,
        zh_gsdltp_dir=gsdltp,
    )[0]
    assert record.system == "jp"
    assert record.edits[0].mapped_label == mapped


def test_jp_errant_refuses_unattested_transformations(tmp_path: Path) -> None:
    spec = dataset_spec("en")
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    with pytest.raises(AdapterUnavailableError, match="transformation manifest"):
        run_jp_errant(
            [],
            spec,
            command_prefix=(sys.executable,),
            patch_manifest=tmp_path / "missing.json",
            stanza_resources=stanza,
        )


def test_jp_errant_runs_external_producer_in_private_working_directory(
    tmp_path: Path,
) -> None:
    script = tmp_path / "jp-cwd.py"
    script.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "assert Path.cwd().resolve() == Path(a.out).parent.resolve()\n"
        "Path('error_analysis_log.txt').write_text('isolated', encoding='utf-8')\n"
        "s=Path(a.orig).read_text(encoding='utf-8').strip()\n"
        "Path(a.out).write_text('S '+s+'\\nA 0 1|||R:NOUN -> NOUN|||b|||"
        "REQUIRED|||-NONE-|||0\\n\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("en")
    pair = PairRecord(spec.dataset_id, "0", "a", "b", None, None)

    [record] = run_jp_errant(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )

    assert record.edits[0].replacement == "b"
    assert not (tmp_path / "error_analysis_log.txt").exists()


def test_jp_paper_compat_keeps_raw_and_uses_lowest_duplicate_source_id(
    tmp_path: Path,
) -> None:
    script = tmp_path / "jp.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "ss=open(a.orig,encoding='utf-8').read().splitlines()\n"
        "cs=open(a.cor,encoding='utf-8').read().splitlines()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  for s,c in zip(ss,cs):\n"
        "    out.write('S '+s+'\\n')\n"
        "    if s==c: out.write('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\\n')\n"
        "    else: out.write('A 1 2|||R:VERB -> VERB|||ging|||REQUIRED|||-NONE-|||0\\n')\n"
        "    out.write('\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("de")
    pairs = [
        PairRecord(
            spec.dataset_id,
            "1",
            "Er geht",
            "Er ging",
            ("Er", "geht"),
            ("Er", "ging"),
        ),
        PairRecord(
            spec.dataset_id,
            "0",
            "Er geht",
            "Er geht",
            ("Er", "geht"),
            ("Er", "geht"),
        ),
    ]
    views = run_jp_errant_views(
        pairs,
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )
    assert [len(record.edits) for record in views.raw] == [1, 0]
    assert [len(record.edits) for record in views.paper_compat] == [0, 0]


def test_jp_raw_de_does_not_compute_numeric_id_paper_projection(tmp_path: Path) -> None:
    script = _jp_script(tmp_path / "jp.py")
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("de")
    pair = PairRecord(
        spec.dataset_id,
        '"non-numeric"',
        "in Haus",
        "im Haus",
        ("in", "Haus"),
        ("im", "Haus"),
    )
    record = run_jp_errant(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )[0]
    assert record.item_id == '"non-numeric"'
    assert (record.edits[0].start, record.edits[0].end) == (0, 1)


def test_jp_ko_paper_compat_projects_only_frozen_legacy_span(tmp_path: Path) -> None:
    script = tmp_path / "jp-ko.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "s=open(a.orig,encoding='utf-8').read().strip()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  out.write('S '+s+'\\n')\n"
        "  out.write('A 3 4|||R:VERB+ADP -> VERB+ADP|||원해요|||REQUIRED|||"
        "-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("ko")
    pair = PairRecord(
        spec.dataset_id,
        "823",
        "그래서 Ph .D.를 원하요 .",
        "그래서 Ph .D.를 원해요 .",
        ("그래서", "Ph", ".D.를", "원하요", "."),
        ("그래서", "Ph", ".D.를", "원해요", "."),
    )
    views = run_jp_errant_views(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )
    assert (views.raw[0].edits[0].start, views.raw[0].edits[0].end) == (3, 4)
    assert (
        views.paper_compat[0].edits[0].start,
        views.paper_compat[0].edits[0].end,
    ) == (4, 5)


def test_compatibility_cli_flags_cannot_overwrite_raw_or_use_bulk(
    tmp_path: Path,
) -> None:
    same = tmp_path / "same.jsonl"
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(SystemExit):
        gerrant_main(
            [
                "--pairs",
                str(missing),
                "--output",
                str(same),
                "--paper-compat",
                "--raw-output",
                str(same),
            ]
        )
    with pytest.raises(SystemExit):
        jp_main(
            [
                "--dataset",
                "ko",
                "--pairs",
                str(missing),
                "--output",
                str(tmp_path / "paper.jsonl"),
                "--paper-compat",
                "--raw-output",
                str(tmp_path / "raw.jsonl"),
                "--bulk-nonpaper",
            ]
        )


def test_m2_position_alignment_rejects_same_count_wrong_source() -> None:
    spec = dataset_spec("en")
    pair = PairRecord(spec.dataset_id, "0", "expected text", "corrected text")
    with pytest.raises(AdapterOutputError, match="does not match canonical pair"):
        align_m2_by_position([pair], [M2Sentence(("different", "text"), ())])


def test_jp_czech_allows_stanza_kdybych_parser_normalization(tmp_path: Path) -> None:
    script = tmp_path / "jp-cs.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  out.write('S Bylo by krásné , když bych v Čechach pro kurz .\\n')\n"
        "  out.write('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("cs")
    tokens = (
        "Bylo",
        "by",
        "krásné",
        ",",
        "kdybych",
        " ",
        "v",
        "Čechach",
        "pro",
        "kurz",
        ".",
    )
    pair = PairRecord(
        spec.dataset_id,
        "12",
        " ".join(tokens),
        " ".join(tokens),
        tokens,
        tokens,
    )
    record = run_jp_errant(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )[0]
    assert record.item_id == "12"
    assert record.edits == ()


def test_uk_insertion_inside_token_uses_containing_token_boundary() -> None:
    spec = dataset_spec("uk")
    pair = PairRecord(
        spec.dataset_id,
        "0",
        "abcd next",
        "abcd next",
        ("abcd", "next"),
        ("abcd", "next"),
    )
    sentence = M2Sentence(
        ("ab", "cd", "next"),
        (M2Edit(1, 1, "M:PUNCT", ","),),
    )
    legacy_next = normalized_m2_edits(
        pair,
        sentence,
        span_unit="token",
        label_mapper=lambda label: label,
        tool_separator=" ",
    )[0]
    fresh_uk = normalized_m2_edits(
        pair,
        sentence,
        span_unit="token",
        label_mapper=lambda label: label,
        tool_separator=" ",
        insertion_inside_token_uses_containing_boundary=True,
    )[0]
    assert (legacy_next.start, legacy_next.end) == (1, 1)
    assert (fresh_uk.start, fresh_uk.end) == (0, 0)


def test_jp_uk_paper_compat_projects_only_item_1112_legacy_boundary(
    tmp_path: Path,
) -> None:
    script = tmp_path / "jp-uk.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('-orig'); p.add_argument('-cor'); "
        "p.add_argument('-out'); p.add_argument('-lang'); a=p.parse_args()\n"
        "s=open(a.orig,encoding='utf-8').read().strip()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  out.write('S '+s+'\\n')\n"
        "  out.write('A 76 76|||M:PUNCT|||,|||REQUIRED|||-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("uk")
    source_tokens = [f"t{index}" for index in range(80)]
    source_tokens[75:78] = ["зайвим", "роботі", "розуму"]
    correction_tokens = list(source_tokens)
    correction_tokens[75] = "зайвим,"
    pair = PairRecord(
        spec.dataset_id,
        "1112",
        " ".join(source_tokens),
        " ".join(correction_tokens),
        tuple(source_tokens),
        tuple(correction_tokens),
    )
    views = run_jp_errant_views(
        [pair],
        spec,
        command_prefix=(sys.executable, script),
        patch_manifest=manifest,
        stanza_resources=stanza,
    )
    assert (views.raw[0].edits[0].start, views.raw[0].edits[0].end) == (76, 76)
    assert (
        views.paper_compat[0].edits[0].start,
        views.paper_compat[0].edits[0].end,
    ) == (77, 77)


def test_jp_sharded_driver_preserves_input_order(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('--mode'); p.add_argument('-orig'); "
        "p.add_argument('-cor'); p.add_argument('-out'); p.add_argument('-lang'); "
        "a=p.parse_args()\n"
        "ss=open(a.orig,encoding='utf-8').read().splitlines()\n"
        "with open(a.out,'w',encoding='utf-8') as out:\n"
        "  for s in ss:\n"
        "    out.write('S '+s+'\\n')\n"
        "    out.write('A 0 1|||R:NOUN -> NOUN|||'+s.split()[0]+"
        "'|||REQUIRED|||-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "jp.json", commit=JP_COMMIT, patches=JP_PATCHES
    )
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    spec = dataset_spec("de")
    pairs = [
        PairRecord(
            spec.dataset_id,
            str(index),
            f"Wort{index} alt",
            f"Wort{index} neu",
            (f"Wort{index}", "alt"),
            (f"Wort{index}", "neu"),
        )
        for index in range(4)
    ]
    views = run_jp_errant_sharded_views(
        pairs,
        spec,
        python=sys.executable,
        worker=worker,
        workers=2,
        patch_manifest=manifest,
        stanza_resources=stanza,
    )
    assert [record.item_id for record in views.raw] == ["0", "1", "2", "3"]
    assert [record.edits[0].replacement for record in views.raw] == [
        "Wort0",
        "Wort1",
        "Wort2",
        "Wort3",
    ]


def test_jp_bulk_mode_requires_explicit_nonpaper_override(tmp_path: Path) -> None:
    with pytest.raises(AdapterOutputError, match="bulk_process changes"):
        run_jp_errant_bulk(
            [],
            dataset_spec("en"),
            python=sys.executable,
            patch_manifest=tmp_path / "unused.json",
            stanza_resources=tmp_path,
        )


def test_cherrant_gold_runs_in_isolated_pinned_environment(tmp_path: Path) -> None:
    root = tmp_path / "cherrant"
    root.mkdir()
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    script = root / "parallel_to_m2.py"
    script.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser(); p.add_argument('--file'); p.add_argument('--output'); "
        "p.add_argument('--granularity'); a=p.parse_args()\n"
        "rows=open(a.file,encoding='utf-8').read().splitlines()\n"
        "with open(a.output,'w',encoding='utf-8') as out:\n"
        "  for row in rows:\n"
        "    _,src,_=row.split('\\t')\n"
        "    out.write('S '+' '.join(src)+'\\n')\n"
        "    out.write('A 0 1|||R:NOUN|||您|||REQUIRED|||-NONE-|||0\\n\\n')\n",
        encoding="utf-8",
    )
    manifest = _write_patch_manifest(
        tmp_path / "cherrant.json",
        commit=CHERRANT_COMMIT,
        patches=CHERRANT_PATCHES,
        ltp_model_revision=EXPECTED_LTP_REVISION,
    )
    spec = dataset_spec("zh")
    pair = PairRecord(spec.dataset_id, '"a"', "你好", "您好")
    record = run_cherrant_reference(
        [pair],
        spec,
        python=sys.executable,
        entrypoint=script,
        cherrant_root=root,
        patch_manifest=manifest,
        huggingface_cache=cache,
    )[0]
    assert record.system == "gold"
    assert record.edits[0].mapped_label == "NOUN"
    assert (record.edits[0].start, record.edits[0].end) == (0, 1)
