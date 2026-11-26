"""Pinned provenance and preflight metadata for the METC paper systems.

The command templates preserve the evaluated systems' historical interfaces.
They are not the release execution API. Canonical hydrated execution,
materialization, and fresh-versus-frozen comparison live in
``metc_artifact.e2e`` and are exposed by ``reproduce.py``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from metc_artifact.schema import DATASETS, dataset_spec

from .models import (
    AssetKind,
    AssetRequirement,
    CommandTemplate,
    DatasetBinding,
    EnvironmentRequirement,
    Provenance,
    RuntimeEnvironment,
    SystemRole,
    SystemSpec,
)


ALL_DATASETS = tuple(dataset.dataset_id for dataset in DATASETS)
ENGLISH_DATASETS = ("en_conll14_ann0", "en_conll14_ann1")


def _bindings(*dataset_ids: str, **values: str) -> tuple[DatasetBinding, ...]:
    command_values = tuple(sorted(values.items()))
    return tuple(DatasetBinding(dataset_id, command_values) for dataset_id in dataset_ids)


_SYSTEMS = (
    SystemSpec(
        system_id="metc",
        annotation_key="metc",
        display_name="METC student",
        role=SystemRole.ANNOTATOR,
        runtime=RuntimeEnvironment(
            name="metc-paper-py312-cuda",
            executable_environment_variable="METC_SYSTEM_PYTHON",
            default_executable=".venv/bin/python",
            python_version="3.12",
            dependencies=(
                "torch==2.9.0",
                "transformers==4.57.3",
                "tokenizers==0.22.0",
                "safetensors==0.6.2",
                "flash-attn==2.8.2",
            ),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "-m",
                "metc.cli",
                "classify",
                "--model",
                "{checkpoint}",
                "--input",
                "{input}",
                "--output",
                "{output}",
                "--device",
                "cuda",
                "--inference-profile",
                "paper",
            ),
            required_values=("input", "output"),
            input_files=("input",),
        ),
        bindings=_bindings(*ALL_DATASETS),
        assets=(
            AssetRequirement(
                name="checkpoint",
                kind=AssetKind.DIRECTORY,
                description="extracted and hydrated METC checkpoint",
                environment_variable="METC_MODEL_PATH",
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/superhuman/metc/releases/tag/v1.0.0",
            revision="v1.0.0",
            package_version="1.0.0",
            license="Apache-2.0",
            evaluated_configuration=(
                ("architecture", "ModernBertForTokenClassification"),
                ("labels", "42"),
                ("parameters", "140658474"),
                ("checkpoint_archive_sha256", "06ddc2bc1425187bc358785fc3af5febba910ee8c5a38e8b2b7be44cf905f8a3"),
                ("base_tokenizer", "jhu-clsp/mmBERT-small"),
                ("base_tokenizer_revision", "abc32620dd4f6ab06f5fbe905dc25f310618e09f"),
                ("max_sequence_length", "512"),
                ("padding", "max_length"),
                ("padding_side", "right"),
                ("pretokenization_character_limit", "3584"),
                ("truncation_side", "right"),
                ("truncated_edit_fallback", "Other/Compound_Complex"),
                ("precision", "bfloat16"),
                ("attention", "flash_attention_2"),
                ("cleanupSemantic", "false"),
                ("ner", "true"),
                ("omnilingual", "true"),
                ("uk_tokenized", "true"),
                ("other_settings_tokenized", "false"),
            ),
            historical_commands=(
                "python data_prep/regen_metc_all.py --fetch --languages <codes> --workers 24",
                "python data_prep/regen_metc_all.py --apply --languages <codes>",
            ),
        ),
        availability_note=(
            "The canonical E2E METC adapter used by reproduce.py bundles pair construction, "
            "invokes the classifier, materializes annotations, and compares them with the "
            "frozen Data artifact."
        ),
    ),
    SystemSpec(
        system_id="metc_teacher",
        annotation_key="metc_teacher",
        display_name="METC teacher",
        role=SystemRole.TEACHER,
        runtime=RuntimeEnvironment(
            name="metc-teacher-direct-openai",
            executable_environment_variable="METC_TEACHER_PYTHON",
            default_executable=".venv/bin/python",
            python_version="3.12",
            required_environment=(
                EnvironmentRequirement(
                    name="OpenAI API credentials",
                    any_of=("OPENAI_API_KEY",),
                    description="credential used by the bundled direct OpenAI client",
                    secret=True,
                ),
            ),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "-m",
                "metc_artifact.judges.batch",
                "run-teacher",
                "--dataset",
                "{dataset_id}",
                "--benchmark",
                "{benchmark}",
                "--output-root",
                "{output_root}",
                "--run-id",
                "{run_id}",
            ),
            required_values=("benchmark", "output_root", "run_id"),
            input_files=("benchmark",),
        ),
        bindings=_bindings(*ALL_DATASETS),
        assets=(
            AssetRequirement(
                name="taxonomy",
                kind=AssetKind.FILE,
                description="released METC annotation taxonomy",
                default_relative_path="prompts/taxonomy_v2.md",
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/superhuman/metc/releases/tag/v1.0.0",
            revision="v1.0.0",
            local_modifications=(
                "The bundled driver relabels immutable fresh METC spans and replacements.",
            ),
            evaluated_configuration=(
                ("model", "gpt-5.4"),
                ("reasoning_effort", "low"),
                ("system_message", "Return only valid JSON per the output contract. No extra text."),
            ),
        ),
        availability_note=(
            "The canonical E2E teacher adapter bundles the annotation prompt, taxonomy, direct "
            "OpenAI client, stable-ID job builder, and materializer; fresh execution requires "
            "OPENAI_API_KEY."
        ),
    ),
    SystemSpec(
        system_id="errant",
        annotation_key="errant",
        display_name="ERRANT English",
        role=SystemRole.ANNOTATOR,
        runtime=RuntimeEnvironment(
            name="errant-en-py311",
            executable_environment_variable="ERRANT_PYTHON",
            python_version="3.11.13",
            dependencies=(
                "errant==3.0.0",
                "spacy==3.8.14",
                "en_core_web_sm==3.8.0",
                "rapidfuzz==3.14.5",
            ),
        ),
        command=CommandTemplate(argv=("{executable}", "{entrypoint}", "--apply")),
        bindings=_bindings(*ENGLISH_DATASETS),
        assets=(
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="authoritative regen_errant_en.py producer",
                environment_variable="ERRANT_ENTRYPOINT",
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/chrisjbryant/errant.git",
            revision="4a36d9bf7fa1d6109bea79b23fccc5d7731fabc1",
            package_version="3.0.0",
            license="MIT",
            historical_commands=(
                ".venvs/errant/bin/python experiments/data_prep/regen_errant_en.py --apply",
            ),
        ),
        availability_note=(
            "The adapter is bundled; one-command hydration installs the pinned ERRANT runtime "
            "and English spaCy model."
        ),
    ),
    SystemSpec(
        system_id="gerrant",
        annotation_key="ger",
        display_name="German gERRANT",
        role=SystemRole.ANNOTATOR,
        runtime=RuntimeEnvironment(
            name="gerrant-de-py311",
            executable_environment_variable="GERRANT_PYTHON",
            python_version="3.11.13",
            dependencies=(
                "spacy==3.8.14",
                "de_core_news_sm==3.8.0",
                "nltk==3.9.4",
                "regex==2026.6.28",
            ),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "-orig",
                "{source}",
                "-cor",
                "{correction}",
                "-out",
                "{output}",
            ),
            required_values=("source", "correction", "output"),
            input_files=("source", "correction"),
            working_directory="{gerrant_root}",
        ),
        bindings=_bindings("de_merlin"),
        assets=(
            AssetRequirement(
                name="gerrant_root",
                kind=AssetKind.DIRECTORY,
                description="bundled spaCy 3 gERRANT port",
                default_relative_path="baselines/gerrant_de",
            ),
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="bundled gerrant_de.py",
                default_relative_path="baselines/gerrant_de/gerrant_de.py",
            ),
            AssetRequirement(
                name="resources",
                kind=AssetKind.DIRECTORY,
                description="prepared de_DE-large.txt and de-stts_map resources",
                environment_variable="GERRANT_RESOURCES",
                default_relative_path="baselines/gerrant_de/resources",
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/adrianeboyd/errant",
            revision="92a86d0722536f4b2c6a4a10037b64f3955ad1b5",
            license="MIT",
            local_modifications=(
                "Ported the Boyd German extensions to spaCy 3.",
                "Runtime resources are prepared separately and hash checked.",
                "Historical source-only materialization excludes one duplicate-source edit.",
            ),
            evaluated_configuration=(
                ("native_m2_edits", "2880"),
                ("materialized_evaluation_edits", "2879"),
                ("m2_sha256", "3dc3ce3c59d97b37c6b762981850f5504abdce3a34962fe27ddd7f58105fb4f7"),
                ("de_stts_map_sha256", "840a7d37c2364b3a5f628900a58c4cc513014d47de6d3a96414fa43c9231c869"),
                ("de_DE_large_sha256", "577c06ebd6461c0a43486a9a538c1175fbc1f0d1a3985e9b4350a5ad1c5d9da6"),
            ),
            historical_commands=(
                ".venvs/errant/bin/python experiments/data_prep/regen_gerrant_de.py --apply",
            ),
        ),
        availability_note=(
            "The bundled adapter implements the exact 2,879-edit paper materialization; "
            "one-command hydration installs its pinned runtime and verifies both resources."
        ),
    ),
    SystemSpec(
        system_id="jp_errant",
        annotation_key="jp",
        display_name="JP-ERRANT",
        role=SystemRole.ANNOTATOR,
        runtime=RuntimeEnvironment(
            name="jp-errant-py311",
            executable_environment_variable="JP_ERRANT_PARALLEL",
            python_version="3.11.13",
            dependencies=(
                "jp_errant==3.0.1",
                "stanza==1.10.1",
                "torch==2.2.2",
                "transformers==4.49.0",
                "numpy==1.26.4",
                "scipy==1.15.2",
                "pandas==3.0.3",
                "pypinyin==0.53.0",
            ),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "-orig",
                "{source}",
                "-cor",
                "{correction}",
                "-out",
                "{output}",
                "-lang",
                "{language_code}",
            ),
            required_values=("source", "correction", "output"),
            input_files=("source", "correction"),
        ),
        bindings=(
            DatasetBinding("en_conll14_ann0", (("language_code", "en"),)),
            DatasetBinding("en_conll14_ann1", (("language_code", "en"),)),
            DatasetBinding("de_merlin", (("language_code", "de"),)),
            DatasetBinding("cs_merlin", (("language_code", "cs"),)),
            DatasetBinding("ko_kolla", (("language_code", "ko"),)),
            DatasetBinding("uk_uagec", (("language_code", "uk"),)),
            DatasetBinding("zh_mucgec", (("language_code", "zh"),)),
        ),
        assets=(
            AssetRequirement(
                name="stanza_resources",
                kind=AssetKind.DIRECTORY,
                description="pinned Stanza models for the selected language",
                environment_variable="JP_ERRANT_STANZA_RESOURCES",
            ),
            AssetRequirement(
                name="zh_gsdltp_model",
                kind=AssetKind.DIRECTORY,
                description="GSDLTP Stanza tokenize and POS models used for Chinese",
                environment_variable="ZH_GSDLTP_DIR",
                dataset_ids=("zh_mucgec",),
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/open-writing-evaluation/jp_errant_bea.git",
            revision="839ac8641b15272c2fcb6bc74015e88dda4a8842",
            package_version="3.0.1",
            license="not declared upstream",
            local_modifications=(
                "Chinese loads explicit GSDLTP Stanza tokenize and POS models.",
                "parallel_to_m2 forces legacy=True.",
                "Missing private Chinese glyph table uses an empty-table fallback.",
            ),
            evaluated_configuration=(
                ("legacy", "true"),
                ("zh_model_doi", "10.5281/zenodo.15096936"),
                ("zh_private_glyph_table", "absent; empty-table fallback"),
            ),
            historical_commands=(
                "jp_errant_parallel -orig <orig> -cor <cor> -out <m2> -lang <code>",
            ),
        ),
        availability_note=(
            "The adapter and deterministic transformation recipe are bundled; hydration obtains "
            "the pinned public checkout, verifies the transformed files, and downloads the "
            "required language models."
        ),
    ),
    SystemSpec(
        system_id="conll14_reference",
        annotation_key="gold",
        display_name="CoNLL-2014 reference constructor",
        role=SystemRole.REFERENCE_CONSTRUCTOR,
        runtime=RuntimeEnvironment(
            name="conll14-reference-isolated",
            executable_environment_variable="CONLL14_REFERENCE_PYTHON",
            python_version=None,
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "--dataset",
                "{experiment_code}",
                "--input",
                "{input}",
                "--output",
                "{output}",
            ),
            required_values=("input", "output"),
            input_files=("input",),
        ),
        bindings=_bindings(*ENGLISH_DATASETS),
        assets=(
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="restored CoNLL M2 reference-construction adapter",
                environment_variable="CONLL14_REFERENCE_ENTRYPOINT",
            ),
        ),
        provenance=Provenance(
            repository=None,
            revision=None,
            evaluated_configuration=(
                ("archive_sha256", "d45784ef928b47521d2fb30f4fa24e59d316357f1567eae922ce02560ba6cf04"),
                ("ann0_m2", "noalt/official-2014.0.m2"),
                ("ann1_m2", "noalt/official-2014.1.m2"),
            ),
            historical_commands=("python experiments/run_english_alignment.py",),
        ),
        availability_note=(
            "Implemented in the bundled public-dataset reconstruction path, which downloads the "
            "official CoNLL-2014 archive and verifies both annotator references."
        ),
    ),
    SystemSpec(
        system_id="merlin_reference",
        annotation_key="gold",
        display_name="MERLIN reference constructor",
        role=SystemRole.REFERENCE_CONSTRUCTOR,
        runtime=RuntimeEnvironment(
            name="merlin-reference-isolated",
            executable_environment_variable="MERLIN_REFERENCE_PYTHON",
            python_version=None,
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "--dataset",
                "{experiment_code}",
                "--input",
                "{input}",
                "--output",
                "{output}",
            ),
            required_values=("input", "output"),
            input_files=("input",),
        ),
        bindings=_bindings("de_merlin", "cs_merlin"),
        assets=(
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="MERLIN EXMARaLDA reference-construction adapter",
                environment_variable="MERLIN_REFERENCE_ENTRYPOINT",
            ),
        ),
        provenance=Provenance(
            repository=None,
            revision="MERLIN corpus version 1.1",
            local_modifications=(
                "German selects the 103 public Boyd (W-NUT 2018) MERLIN development essay IDs.",
                "Czech retains source/TH1 sentence pairs from all 441 files; 439 are usable.",
            ),
            evaluated_configuration=(
                ("de_sentences", "1197"),
                ("cs_sentences", "6590"),
            ),
        ),
        availability_note=(
            "Implemented in the bundled public-dataset reconstruction path for the pinned MERLIN "
            "release and public German selection manifest."
        ),
    ),
    SystemSpec(
        system_id="kolla_reference",
        annotation_key="gold",
        display_name="KoLLA/KAGAS-derived reference constructor",
        role=SystemRole.REFERENCE_CONSTRUCTOR,
        runtime=RuntimeEnvironment(
            name="kolla-reference-isolated",
            executable_environment_variable="KOLLA_REFERENCE_PYTHON",
            python_version=None,
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "--input",
                "{input}",
                "--output",
                "{output}",
            ),
            required_values=("input", "output"),
            input_files=("input",),
        ),
        bindings=_bindings("ko_kolla"),
        assets=(
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="KoLLA annotator-0 M2 reference-construction adapter",
                environment_variable="KOLLA_REFERENCE_ENTRYPOINT",
            ),
        ),
        provenance=Provenance(
            repository=None,
            revision=None,
            local_modifications=(
                "Uses annotator 0 only and preserves corrected same-position multi-edit handling.",
            ),
            evaluated_configuration=(
                ("source_file", "KoLLA_multi-refs.m2"),
                ("sentences", "1418"),
                ("edits", "1768"),
            ),
        ),
        availability_note=(
            "Implemented in the bundled public-dataset reconstruction path for the pinned KoLLA "
            "annotator-0 M2 release."
        ),
    ),
    SystemSpec(
        system_id="uagec_reference",
        annotation_key="gold",
        display_name="UA-GEC reference constructor",
        role=SystemRole.REFERENCE_CONSTRUCTOR,
        runtime=RuntimeEnvironment(
            name="uagec-reference-isolated",
            executable_environment_variable="UAGEC_REFERENCE_PYTHON",
            python_version=None,
            dependencies=("ua_gec==2.1.3",),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "--input",
                "{input}",
                "--output",
                "{output}",
            ),
            required_values=("input", "output"),
            input_files=("input",),
        ),
        bindings=_bindings("uk_uagec"),
        assets=(
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="UA-GEC paragraph/reference-construction adapter",
                environment_variable="UAGEC_REFERENCE_ENTRYPOINT",
            ),
        ),
        provenance=Provenance(
            repository=None,
            revision="ua_gec==2.1.3",
            package_version="2.1.3",
            local_modifications=(
                "Builds the test/gec-only paragraph set and backfills exact replacements from .ann.",
            ),
            evaluated_configuration=(
                ("partition", "test"),
                ("annotation_layer", "gec-only"),
                ("sentences", "2127"),
                ("edits", "5390"),
            ),
        ),
        availability_note=(
            "Implemented in the bundled public-dataset reconstruction path for the pinned UA-GEC "
            "test/gec-only partition."
        ),
    ),
    SystemSpec(
        system_id="cherrant_reference",
        annotation_key="gold",
        display_name="MuCGEC ChERRANT reference constructor",
        role=SystemRole.REFERENCE_CONSTRUCTOR,
        runtime=RuntimeEnvironment(
            name="cherrant-ltp-isolated",
            executable_environment_variable="CHERRANT_PYTHON",
            python_version="3.11.13",
            dependencies=(
                "huggingface-hub==0.22.2",
                "ltp-core==0.1.4",
                "ltp==4.2.14",
                "opencc==1.3.1",
                "pypinyin==0.55.0",
                "python-levenshtein==0.12.1",
                "torch==2.12.1",
                "transformers==4.40.0",
            ),
        ),
        command=CommandTemplate(
            argv=(
                "{executable}",
                "{entrypoint}",
                "--file",
                "{input}",
                "--output",
                "{output}",
                "--granularity",
                "word",
            ),
            required_values=("input", "output"),
            input_files=("input",),
            working_directory="{cherrant_root}",
        ),
        bindings=_bindings("zh_mucgec"),
        assets=(
            AssetRequirement(
                name="cherrant_root",
                kind=AssetKind.DIRECTORY,
                description="pinned MuCGEC scorers/ChERRANT checkout",
                environment_variable="CHERRANT_ROOT",
            ),
            AssetRequirement(
                name="entrypoint",
                kind=AssetKind.FILE,
                description="patched ChERRANT parallel_to_m2.py",
                environment_variable="CHERRANT_ENTRYPOINT",
            ),
            AssetRequirement(
                name="ltp_model",
                kind=AssetKind.DIRECTORY,
                description="pinned LTP/small model snapshot",
                environment_variable="CHERRANT_LTP_MODEL",
            ),
        ),
        provenance=Provenance(
            repository="https://github.com/HillZhang1999/MuCGEC",
            revision="39221934ee4ccf07801879136a75dd7e11319977",
            local_modifications=(
                "Tokenizer lazily imports LTP and uses the LTP 4.2 pipeline API with LTP/small.",
                "The first MuCGEC development reference is used and sentinel targets are excluded.",
            ),
            evaluated_configuration=(
                ("granularity", "word"),
                ("ltp_model", "LTP/small"),
                ("ltp_model_revision", "dea87bf0da3c5d054bbd426a78d7fa9efd347799"),
                ("raw_dev_rows", "1137"),
                ("final_rows", "1079"),
                ("final_edits", "4191"),
            ),
            historical_commands=(
                "python parallel_to_m2.py -f <id-src-ref1.para> -o <gold.m2> -g word",
            ),
        ),
        availability_note=(
            "Implemented in the bundled public-dataset reconstruction path; one-command hydration "
            "obtains the pinned MuCGEC checkout and exact ChERRANT/LTP runtime."
        ),
    ),
)


SYSTEMS: Mapping[str, SystemSpec] = MappingProxyType(
    {system.system_id: system for system in _SYSTEMS}
)


def system_spec(system_id: str) -> SystemSpec:
    try:
        return SYSTEMS[system_id]
    except KeyError as error:
        raise ValueError(f"unknown METC annotation producer: {system_id!r}") from error


def systems_for_dataset(identifier: str) -> tuple[SystemSpec, ...]:
    canonical_id = dataset_spec(identifier).dataset_id
    return tuple(system for system in _SYSTEMS if canonical_id in system.dataset_ids)
