"""MuCGEC/ChERRANT reference generation and conversion."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .m2 import M2Edit, parse_m2


MUGEC_DEV_SHA256 = "edbe29627aa8c61b1196b0b8c85898560dd2218d5031857921e9b621e5241e18"
UPSTREAM_TOKENIZER_SHA256 = "e629e19122e8938d3759f739e30c8244c375ed6af96b476af53d886c8bba8d74"
PARALLEL_TO_M2_SHA256 = "25cf6b1e04dc79272aa2799fe591e2a37ebad4cd2c3e314dc0fa9748c34ed9f7"
REFERENCE_M2_SHA256 = "509c0296f3910f37793c62a9376c68812a07ab5826135277011379110289d14f"
LTP_MODEL_REVISION = "dea87bf0da3c5d054bbd426a78d7fa9efd347799"
CHERRANT_ENVIRONMENT = (
    "python==3.11",
    "ltp==4.2.14",
    "ltp-core==0.1.4",
    "torch==2.12.1",
    "transformers==4.40.0",
    "opencc==1.3.1",
    "python-Levenshtein==0.12.1",
    "pypinyin==0.55.0",
    "huggingface-hub==0.22.2",
)

_CATEGORIES = {
    "VERB",
    "NOUN",
    "PUNCT",
    "AUX",
    "ADV",
    "PREP",
    "PRON",
    "CONJ",
    "ADJ",
    "NUM",
    "QUAN",
    "SPELL",
    "OTHER",
    "W",
}

_PATCHED_TOKENIZER = '''from typing import List
from pypinyin import pinyin, Style, lazy_pinyin
import torch
import os
import functools

class Tokenizer:
    """ChERRANT tokenizer with the LTP 4.2 API used for the paper."""

    def __init__(self, granularity: str = "word", device: str = "cpu",
                 segmented: bool = False, bpe: bool = False) -> None:
        self.ltp = None
        if granularity == "word":
            from ltp import LTP
            self.ltp = LTP("LTP/small@dea87bf0da3c5d054bbd426a78d7fa9efd347799")
            try:
                self.ltp.add_words(words=["[缺失成分]"])
            except Exception:
                pass
        self.segmented = segmented
        self.granularity = granularity
        if self.granularity == "word":
            self.tokenizer = self.split_word
        elif self.granularity == "char":
            self.tokenizer = functools.partial(self.split_char, bpe=bpe)
        else:
            raise NotImplementedError

    def __call__(self, input_strings: List[str]) -> List:
        if not self.segmented:
            input_strings = ["".join(s.split(" ")) for s in input_strings]
        return self.tokenizer(input_strings)

    def split_char(self, input_strings: List[str], bpe=False) -> List:
        if bpe:
            from . import tokenization
            project_dir = os.path.dirname(os.path.dirname(__file__))
            tokenizer = tokenization.FullTokenizer(
                vocab_file=os.path.join(project_dir, "data", "chinese_vocab.txt"),
                do_lower_case=False,
            )
        results = []
        for input_string in input_strings:
            if not self.segmented:
                segment_string = " ".join(
                    [char for char in input_string]
                    if not bpe else tokenizer.tokenize(input_string)
                )
            else:
                segment_string = input_string
            segment_string = segment_string.replace(
                "[ 缺 失 成 分 ]", "[缺失成分]"
            ).split(" ")
            results.append([
                (char, "unk", pinyin(char, style=Style.NORMAL, heteronym=True)[0])
                for char in segment_string
            ])
        return results

    def split_word(self, input_strings: List[str]) -> List:
        texts = ["".join(s.split(" ")) if self.segmented else s for s in input_strings]
        out = self.ltp.pipeline(texts, tasks=["cws", "pos"])
        result = []
        for words, tags in zip(out.cws, out.pos):
            readings = [lazy_pinyin(word) for word in words]
            result.append(list(zip(words, tags, readings)))
        return result
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(root: Path, pattern: str, description: str) -> Path:
    paths = list(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {description} under {root}, found {len(paths)}")
    return paths[0]


def mucgec_development_file(root: Path) -> Path:
    path = _one(root, "MuCGEC_dev.txt", "MuCGEC development file")
    actual = _sha256(path)
    if actual != MUGEC_DEV_SHA256:
        raise ValueError(f"MuCGEC dev checksum mismatch: expected {MUGEC_DEV_SHA256}, got {actual}")
    return path


def _raw_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"MuCGEC row has fewer than three fields: {line!r}")
        rows.append((fields[0], fields[1], fields[2]))
    return rows


def write_first_reference_parallel(root: Path, output: Path) -> Path:
    """Write the exact id/source/first-reference input consumed by ChERRANT."""
    rows = _raw_rows(mucgec_development_file(root))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        "".join(f"{identifier}\t{source}\t{target}\n" for identifier, source, target in rows),
        encoding="utf-8",
    )
    return Path(output)


def generate_cherrant_reference(mucgec_root: Path, work_dir: Path, python_executable: Path) -> Path:
    """Run the pinned word-level ChERRANT recipe and verify its exact M2 output."""
    source_root = _one(mucgec_root, "ChERRANT", "MuCGEC ChERRANT directory")
    if source_root.parent.name != "scorers":
        raise ValueError(f"unexpected ChERRANT location: {source_root}")
    upstream_tokenizer = source_root / "modules" / "tokenizer.py"
    if _sha256(upstream_tokenizer) != UPSTREAM_TOKENIZER_SHA256:
        raise ValueError("pinned MuCGEC checkout has an unexpected ChERRANT tokenizer")
    if _sha256(source_root / "parallel_to_m2.py") != PARALLEL_TO_M2_SHA256:
        raise ValueError("pinned MuCGEC checkout has an unexpected parallel_to_m2.py")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "cherrant_ref_zh.m2"
    parallel_output = work_dir / "mucgec-first-ref.tsv"
    with tempfile.TemporaryDirectory(prefix=".cherrant-run-", dir=work_dir) as temporary:
        run_root = Path(temporary)
        cherrant_root = run_root / "ChERRANT"
        shutil.copytree(
            source_root,
            cherrant_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (cherrant_root / "modules" / "tokenizer.py").write_text(
            _PATCHED_TOKENIZER,
            encoding="utf-8",
        )
        parallel = write_first_reference_parallel(
            mucgec_root,
            run_root / "mucgec-first-ref.tsv",
        )
        candidate = run_root / "cherrant_ref_zh.m2"
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        result = subprocess.run(
            (
                str(python_executable),
                str(cherrant_root / "parallel_to_m2.py"),
                "-f",
                str(parallel),
                "-o",
                str(candidate),
                "-g",
                "word",
                "-s",
                # The frozen reference retains 19 alternative equally-cheap alignments.
                "all",
            ),
            cwd=cherrant_root,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            detail = "\n".join(result.stderr.splitlines()[-20:])
            raise RuntimeError(
                f"ChERRANT failed with exit code {result.returncode}:\n{detail}"
            )
        actual = _sha256(candidate)
        if actual != REFERENCE_M2_SHA256:
            raise RuntimeError(
                f"ChERRANT reference differs from the paper run: expected "
                f"{REFERENCE_M2_SHA256}, got {actual}. Verify "
                f"{', '.join(CHERRANT_ENVIRONMENT)}."
            )
        parallel.replace(parallel_output)
        candidate.replace(output)
    return output


def cherrant_category(label: str) -> str:
    normalized = label.strip()
    if not normalized or normalized.lower() == "noop":
        return ""
    if normalized in {"W", "WO"}:
        return "W"
    category = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized
    category = category.split()[0] if category else category
    return category if category in _CATEGORIES else "OTHER"


def _char_edit(tokens: tuple[str, ...], edit: M2Edit) -> list[object]:
    cumulative = [0]
    for token in tokens:
        cumulative.append(cumulative[-1] + len(token))
    start = min(max(edit.start, 0), len(tokens))
    end = min(max(edit.end, 0), len(tokens))
    return [
        cumulative[start],
        cumulative[end],
        cherrant_category(edit.label),
        edit.label,
        edit.replacement,
    ]


def reconstruct_mucgec(root: Path, reference_m2: Path) -> list[dict[str, object]]:
    """Convert the paper's generated ChERRANT M2 into the Chinese reference rows."""
    if _sha256(reference_m2) != REFERENCE_M2_SHA256:
        raise ValueError("the supplied ChERRANT M2 is not the pinned paper reference")
    raw_rows = _raw_rows(mucgec_development_file(root))
    m2_rows = list(parse_m2(reference_m2))
    if len(raw_rows) != len(m2_rows):
        raise ValueError(f"MuCGEC/ChERRANT row mismatch: {len(raw_rows)} != {len(m2_rows)}")

    rows: list[dict[str, object]] = []
    for (identifier, source, target), sentence in zip(raw_rows, m2_rows, strict=True):
        if target in {"没有错误", "无法标注"}:
            continue
        joined = "".join(sentence.tokens)
        if joined != source:
            raise ValueError(f"ChERRANT source mismatch for MuCGEC item {identifier}")
        edits = [
            edit for edit in sentence.edits if edit.start >= 0 and edit.label.lower() != "noop"
        ]
        rows.append(
            {
                "idx": identifier,
                "src": source,
                "tgt": target,
                "gold": [_char_edit(sentence.tokens, edit) for edit in edits],
            }
        )
    return rows
