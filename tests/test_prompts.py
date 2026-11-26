import importlib.util
from pathlib import Path

from metc.mapping import load_label_mapping
from metc_artifact.judges.protocols import taxonomy_pairs


PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt_module(name: str):
    path = PROMPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_prompt_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_annotation_messages_include_exact_system_message_and_settings():
    prompt = load_prompt_module("annotation_prompt")
    messages = prompt.build_annotation_messages(
        "She go.",
        "She goes.",
        "She {#1 go=>goes}.",
        [{"source": "go", "target": "goes"}],
        taxonomy="SYNTHETIC TAXONOMY",
    )
    assert messages[0] == {
        "role": "system",
        "content": "Return only valid JSON per the output contract. No extra text.",
    }
    assert messages[1]["role"] == "user"
    assert "SYNTHETIC TAXONOMY" in messages[1]["content"]
    assert "1. go => goes" in messages[1]["content"]
    assert prompt.ANNOTATION_REQUEST_SETTINGS == {"reasoning_effort": "low"}


def test_annotation_prompt_embeds_the_released_taxonomy():
    prompt = load_prompt_module("annotation_prompt")
    taxonomy = prompt.load_annotation_taxonomy()
    expected = {f"{label}/{sublabel}" for label, sublabel in load_label_mapping().values()}
    assert taxonomy_pairs(taxonomy) == expected

    rendered = prompt.build_annotation_prompt(
        "She go.",
        "She goes.",
        "She {#1 go=>goes}.",
        [{"source": "go", "target": "goes"}],
    )
    assert taxonomy in rendered


def test_acceptability_request_matches_primary_wire_contract():
    judge = load_prompt_module("acceptability_judge")
    payload = judge.build_judge_request("PROMPT", "GPT-5.4")
    assert payload == {
        "model": "GPT-5.4",
        "messages": [
            {
                "role": "system",
                "content": ("Return only valid JSON per the contract; no markdown, no extra text."),
            },
            {"role": "user", "content": "PROMPT"},
        ],
        "temperature": 0,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    }


def test_acceptability_vendor_payload_uses_only_extra_body_branch():
    judge = load_prompt_module("acceptability_judge")
    payload = judge.build_judge_request(
        "PROMPT", "SECOND", extra_body={"thinking_config": {"level": "low"}}
    )
    assert payload["extra_body"] == {"thinking_config": {"level": "low"}}
    assert "response_format" not in payload
    assert "n" not in payload
