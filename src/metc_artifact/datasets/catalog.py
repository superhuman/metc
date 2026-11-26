"""Pinned public inputs for rebuilding the seven evaluation references.

The catalog distinguishes acquisition from redistribution.  In particular,
CoNLL-2014 is downloaded from its official host because the archive does not
declare a reusable data license.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class LicenseTerms:
    name: str
    terms_url: str
    spdx_id: str | None
    redistribution: Literal["permitted", "source-specific", "official-download-only"]
    notice: str


@dataclass(frozen=True)
class HttpArtifact:
    source_id: str
    url: str
    filename: str
    sha256: str
    archive_kind: Literal["file", "tar.gz", "zip"]
    official_page: str
    version: str
    license: LicenseTerms


@dataclass(frozen=True)
class GitArtifact:
    source_id: str
    url: str
    commit: str
    official_page: str
    version: str
    license: LicenseTerms


SourceArtifact = HttpArtifact | GitArtifact


@dataclass(frozen=True)
class DatasetRecipe:
    dataset_id: str
    source_ids: tuple[str, ...]
    expected_items: int
    expected_gold_edits: int
    note: str


CONLL_TERMS = LicenseTerms(
    name="CoNLL-2014 shared-task terms",
    terms_url="https://www.comp.nus.edu.sg/~nlp/conll14st.html",
    spdx_id=None,
    redistribution="official-download-only",
    notice=(
        "The official archive does not declare a general redistribution license; "
        "acquire it directly from the NUS shared-task host."
    ),
)
MERLIN_TERMS = LicenseTerms(
    name="Creative Commons Attribution-ShareAlike 4.0 International",
    terms_url="https://creativecommons.org/licenses/by-sa/4.0/",
    spdx_id="CC-BY-SA-4.0",
    redistribution="permitted",
    notice="Attribution and ShareAlike conditions apply to the MERLIN corpus.",
)
BOYD_TERMS = LicenseTerms(
    name="Boyd W-NUT 2018 release and source-corpus terms",
    terms_url="https://github.com/adrianeboyd/boyd-wnut2018",
    spdx_id=None,
    redistribution="source-specific",
    notice=(
        "Only the public German development selection IDs are consumed here; "
        "MERLIN text remains governed by CC BY-SA 4.0."
    ),
)
KOLLA_TERMS = LicenseTerms(
    name="GNU General Public License v3.0 or later",
    terms_url="https://zenodo.org/records/15287129",
    spdx_id="GPL-3.0-or-later",
    redistribution="permitted",
    notice="The KoLLA Zenodo record declares GPL-3.0-or-later.",
)
UAGEC_TERMS = LicenseTerms(
    name="Creative Commons Attribution 4.0 International",
    terms_url="https://github.com/grammarly/ua-gec/blob/4757f72f192c4a41e4c8fb1d9690a948f87cf6d6/LICENSE",
    spdx_id="CC-BY-4.0",
    redistribution="permitted",
    notice="Attribution is required for UA-GEC.",
)
MUGEC_TERMS = LicenseTerms(
    name="Apache License 2.0",
    terms_url="https://github.com/HillZhang1999/MuCGEC/blob/39221934ee4ccf07801879136a75dd7e11319977/LICENSE",
    spdx_id="Apache-2.0",
    redistribution="permitted",
    notice="MuCGEC and its bundled ChERRANT code are Apache-2.0 licensed.",
)


_SOURCES: tuple[SourceArtifact, ...] = (
    HttpArtifact(
        source_id="conll14_test",
        url="https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz",
        filename="conll14st-test-data.tar.gz",
        sha256="d45784ef928b47521d2fb30f4fa24e59d316357f1567eae922ce02560ba6cf04",
        archive_kind="tar.gz",
        official_page="https://www.comp.nus.edu.sg/~nlp/conll14st.html",
        version="CoNLL-2014 test release",
        license=CONLL_TERMS,
    ),
    HttpArtifact(
        source_id="merlin_exmaralda_v1_1",
        url=(
            "https://clarin.eurac.edu/repository/xmlui/bitstream/handle/"
            "20.500.12124/6/merlin-exmaralda-v1.1.zip?sequence=8&isAllowed=y"
        ),
        filename="merlin-exmaralda-v1.1.zip",
        sha256="bc3a9dcf167951707d266641ec8e54596ed13957b344f31d57e8a592b96da17c",
        archive_kind="zip",
        official_page="http://hdl.handle.net/20.500.12124/6",
        version="1.1.1 data bundle (EXMARaLDA v1.1)",
        license=MERLIN_TERMS,
    ),
    HttpArtifact(
        source_id="boyd_wnut2018_data",
        url="https://github.com/adrianeboyd/boyd-wnut2018/releases/download/wnut2018/data.tar.gz",
        filename="boyd-wnut2018-data.tar.gz",
        sha256="55210ac71a939d61bea61713bc256c84b4574be7ecce4e64ac3da105968a61bc",
        archive_kind="tar.gz",
        official_page="https://github.com/adrianeboyd/boyd-wnut2018/releases/tag/wnut2018",
        version="wnut2018 (commit 3649c63e6154860e61381b630dc7576dc080acdc)",
        license=BOYD_TERMS,
    ),
    HttpArtifact(
        source_id="kolla_v2_m2",
        url="https://zenodo.org/records/15287129/files/KoLLA_multi-refs.m2?download=1",
        filename="KoLLA_multi-refs.m2",
        sha256="d848bac8095bf4280e9c8f6bb7ef2733434cf0eb2c2977d34fd4a0eb6c061ed4",
        archive_kind="file",
        official_page="https://doi.org/10.5281/zenodo.15287129",
        version="Zenodo record 15287129",
        license=KOLLA_TERMS,
    ),
    GitArtifact(
        source_id="ua_gec",
        url="https://github.com/grammarly/ua-gec.git",
        commit="4757f72f192c4a41e4c8fb1d9690a948f87cf6d6",
        official_page="https://github.com/grammarly/ua-gec",
        version="git commit 4757f72f192c4a41e4c8fb1d9690a948f87cf6d6",
        license=UAGEC_TERMS,
    ),
    GitArtifact(
        source_id="mucgec",
        url="https://github.com/HillZhang1999/MuCGEC.git",
        commit="39221934ee4ccf07801879136a75dd7e11319977",
        official_page="https://github.com/HillZhang1999/MuCGEC",
        version="git commit 39221934ee4ccf07801879136a75dd7e11319977",
        license=MUGEC_TERMS,
    ),
)

SOURCES: Mapping[str, SourceArtifact] = MappingProxyType(
    {source.source_id: source for source in _SOURCES}
)

_RECIPES = (
    DatasetRecipe("en_conll14_ann0", ("conll14_test",), 946, 2_389, "CoNLL annotator 0."),
    DatasetRecipe("en_conll14_ann1", ("conll14_test",), 1_042, 3_207, "CoNLL annotator 1."),
    DatasetRecipe(
        "de_merlin",
        ("merlin_exmaralda_v1_1", "boyd_wnut2018_data"),
        1_197,
        2_412,
        "German MERLIN essays selected by the public Boyd development ID file.",
    ),
    DatasetRecipe(
        "cs_merlin",
        ("merlin_exmaralda_v1_1",),
        6_590,
        14_143,
        "All usable TH1-aligned sentences from the full Czech MERLIN subset.",
    ),
    DatasetRecipe("ko_kolla", ("kolla_v2_m2",), 1_418, 1_768, "KoLLA annotator 0."),
    DatasetRecipe("uk_uagec", ("ua_gec",), 2_127, 5_390, "UA-GEC test/gec-only."),
    DatasetRecipe(
        "zh_mucgec",
        ("mucgec",),
        1_079,
        4_191,
        "MuCGEC dev first reference, annotated by pinned word-level ChERRANT.",
    ),
)

RECIPES: Mapping[str, DatasetRecipe] = MappingProxyType(
    {recipe.dataset_id: recipe for recipe in _RECIPES}
)


def source_artifact(source_id: str) -> SourceArtifact:
    try:
        return SOURCES[source_id]
    except KeyError as error:
        raise ValueError(f"unknown public source: {source_id!r}") from error


def dataset_recipe(dataset_id: str) -> DatasetRecipe:
    try:
        return RECIPES[dataset_id]
    except KeyError as error:
        raise ValueError(f"unknown evaluation dataset: {dataset_id!r}") from error
