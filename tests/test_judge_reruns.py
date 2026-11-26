from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from metc_artifact.judges import (
    GOOGLE_ENDPOINT,
    OPENAI_ENDPOINT,
    FreshRunWriter,
    HTTPResponse,
    compare_with_frozen,
    read_fresh_records,
    run_google_acceptability,
    run_openai_acceptability,
    run_openai_teacher,
)
from metc_artifact.judges.clients import REDACTED_SECRET, SAFE_RESPONSE_HEADERS
from metc_artifact.judges.protocols import ProtocolError, load_teacher_taxonomy


def acceptability_job() -> dict[str, Any]:
    return {
        "edit_id": "metc-edit-v1:de:42:metc:0",
        "edit_idx": 0,
        "row_idx": 42,
        "system": "metc",
        "lang": "German",
        "src": "Er gehen heute.",
        "tgt": "Er geht heute.",
        "edit_diff": "'gehen' => 'geht'",
        "label": "Grammar/VerbSVA",
        "label_gloss": "Problem with subject-verb agreement.",
    }


def openai_response(content: str, *, status: int = 200) -> HTTPResponse:
    body = json.dumps(
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    return HTTPResponse(status=status, body=body, headers={"x-request-id": "request-1"})


def test_openai_primary_uses_exact_public_frozen_settings() -> None:
    calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return openai_response(
            json.dumps(
                {
                    "span_quality": "Good",
                    "label_quality": "Acceptable",
                    "span_reason": "The span isolates gehen.",
                    "label_reason": "The description is valid but broad.",
                }
            )
        )

    record = run_openai_acceptability(
        acceptability_job(), api_key="test-openai-secret", transport=transport
    )

    released_path = Path(__file__).resolve().parents[1] / "prompts" / "acceptability_judge.py"
    released_spec = importlib.util.spec_from_file_location(
        "released_acceptability_prompt", released_path
    )
    assert released_spec is not None and released_spec.loader is not None
    released_prompt = importlib.util.module_from_spec(released_spec)
    released_spec.loader.exec_module(released_prompt)
    assert record["prompt"]["text"] == released_prompt.JUDGE_PROMPT.format(
        language="German",
        source="Er gehen heute.",
        target="Er geht heute.",
        edit_diff="'gehen' => 'geht'",
        label_gloss="Problem with subject-verb agreement.",
    )

    assert len(calls) == 1
    url, headers, payload, timeout = calls[0]
    assert url == OPENAI_ENDPOINT
    assert headers["Authorization"] == "Bearer test-openai-secret"
    assert timeout == 180.0
    assert payload == {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON per the contract; no markdown, no extra text.",
            },
            {"role": "user", "content": record["prompt"]["text"]},
        ],
        "temperature": 0,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
        "service_tier": "flex",
    }
    assert record["error"] is None
    assert record["response"] == {
        "http_status": 200,
        "headers": {"x-request-id": "request-1"},
    }
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-5.4"
    assert record["request"]["configuration"]["historical_backend_id"] == (
        "openai_direct_gpt54_flex"
    )
    assert (
        record["outputs_by_edit_id"][acceptability_job()["edit_id"]]["label_quality"]
        == "Acceptable"
    )
    assert (
        record["raw_response"]
        == openai_response(
            json.dumps(
                {
                    "span_quality": "Good",
                    "label_quality": "Acceptable",
                    "span_reason": "The span isolates gehen.",
                    "label_reason": "The description is valid but broad.",
                }
            )
        ).body
    )
    assert record["prompt"]["sha256"]
    assert "test-openai-secret" not in json.dumps(record)


def test_google_second_uses_native_generate_content_and_records_unknown_thinking() -> None:
    calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        content = json.dumps(
            {
                "span_quality": "Good",
                "label_quality": "Good",
                "span_reason": "The span is exact.",
                "label_reason": "The label precisely describes the edit.",
            }
        )
        return HTTPResponse(
            status=200,
            body=json.dumps({"candidates": [{"content": {"parts": [{"text": content}]}}]}),
            headers={},
        )

    record = run_google_acceptability(
        acceptability_job(), api_key="test-google-secret", transport=transport
    )

    assert len(calls) == 1
    url, headers, payload, _ = calls[0]
    assert url == GOOGLE_ENDPOINT
    assert headers == {
        "Content-Type": "application/json",
        "x-goog-api-key": "test-google-secret",
    }
    assert payload["systemInstruction"] == {
        "parts": [{"text": "Return only valid JSON per the contract; no markdown, no extra text."}]
    }
    assert payload["contents"] == [{"role": "user", "parts": [{"text": record["prompt"]["text"]}]}]
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["responseSchema"]["required"] == [
        "span_quality",
        "label_quality",
        "span_reason",
        "label_reason",
    ]
    assert "thinkingConfig" not in payload["generationConfig"]
    assert record["request"]["endpoint"] == GOOGLE_ENDPOINT
    assert record["request"]["configuration"]["historical_thinking_configuration"] == {
        "status": "not_preserved",
        "fresh_request_behavior": "provider_default",
    }
    assert record["outputs_by_edit_id"][acceptability_job()["edit_id"]]["span_quality"] == "Good"
    assert "test-google-secret" not in json.dumps(record)


def load_released_teacher_prompt_module():
    path = Path(__file__).resolve().parents[1] / "prompts" / "annotation_prompt.py"
    spec = importlib.util.spec_from_file_location("released_annotation_prompt", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openai_teacher_uses_exact_prompt_and_preserves_student_span() -> None:
    calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []
    job = {
        "sentence_id": "de:42",
        "src": "Er gehen heute.",
        "tgt": "Er geht heute.",
        "numbered": "Er {#1 gehen=>geht} heute.",
        "edits": [
            {
                "edit_id": "metc-edit-v1:de:42:metc:0",
                "source": "gehen",
                "target": "geht",
                "span": {"unit": "token", "start": 1, "end": 2},
            }
        ],
    }

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return openai_response(json.dumps({"labels": ["Grammar"], "sublabels": ["VerbSVA"]}))

    record = run_openai_teacher(
        job, api_key="test-openai-secret", transport=transport
    )

    _, _, payload, _ = calls[0]
    assert payload == {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON per the output contract. No extra text.",
            },
            {"role": "user", "content": record["prompt"]["text"]},
        ],
        "reasoning_effort": "low",
        "service_tier": "flex",
    }
    assert "temperature" not in payload
    assert "response_format" not in payload
    released = load_released_teacher_prompt_module()
    taxonomy = released.load_annotation_taxonomy()
    assert record["prompt"]["text"] == released.build_annotation_prompt(
        job["src"],
        job["tgt"],
        job["numbered"],
        [{"source": "gehen", "target": "geht"}],
        taxonomy,
    )
    configuration = record["request"]["configuration"]
    assert configuration["fixed_student_spans"] is True
    assert configuration["taxonomy_sha256"] == hashlib.sha256(taxonomy.encode("utf-8")).hexdigest()
    assert record["input"]["edits"][0]["span"] == {
        "unit": "token",
        "start": 1,
        "end": 2,
    }
    assert record["outputs_by_edit_id"][job["edits"][0]["edit_id"]] == {
        "label": "Grammar",
        "sublabel": "VerbSVA",
        "raw_label": "Grammar/VerbSVA",
    }


def test_fresh_writer_and_stable_id_comparison_never_replace_frozen(tmp_path: Path) -> None:
    def transport(url, headers, payload, timeout):
        return openai_response(
            json.dumps(
                {
                    "span_quality": "Good",
                    "label_quality": "Acceptable",
                    "span_reason": "Exact span.",
                    "label_reason": "Valid broad label.",
                }
            )
        )

    record = run_openai_acceptability(acceptability_job(), api_key="secret", transport=transport)
    writer = FreshRunWriter(tmp_path, "gpt54-20260723")
    path = writer.append("primary", record)
    assert path == tmp_path / "fresh_runs/gpt54-20260723/primary.jsonl"
    loaded = read_fresh_records(path)

    comparison = compare_with_frozen(
        loaded,
        [
            {
                "edit_id": acceptability_job()["edit_id"],
                "span_quality": "Good",
                "label_quality": "Good",
            }
        ],
        fields=("span_quality", "label_quality"),
    )
    assert comparison.frozen_edits == 1
    assert comparison.fresh_edits == 1
    assert comparison.comparable_edits == 1
    assert comparison.field_matches == {"span_quality": 1, "label_quality": 0}
    assert comparison.to_dict()["field_agreement"] == {
        "span_quality": 1.0,
        "label_quality": 0.0,
    }
    assert comparison.mismatches == (
        {
            "edit_id": acceptability_job()["edit_id"],
            "field": "label_quality",
            "frozen": "Good",
            "fresh": "Acceptable",
        },
    )


def test_invalid_provider_content_is_persisted_as_fresh_error() -> None:
    raw = openai_response("not json")

    def transport(url, headers, payload, timeout):
        return raw

    record = run_openai_acceptability(acceptability_job(), api_key="secret", transport=transport)
    assert record["raw_response"] == raw.body
    assert record["provider_content"] == "not json"
    assert record["outputs_by_edit_id"] == {}
    assert record["error"]["type"] == "response_validation_error"


def test_transport_error_is_recorded_without_leaking_credentials() -> None:
    def transport(url, headers, payload, timeout):
        raise RuntimeError("connection failed for test-google-secret")

    record = run_google_acceptability(
        acceptability_job(), api_key="test-google-secret", transport=transport
    )
    assert record["raw_response"] is None
    assert record["response"] == {"http_status": None, "headers": {}}
    assert record["error"] == {
        "type": "transport_error",
        "message": "connection failed for [REDACTED]",
        "status": None,
    }
    assert "test-google-secret" not in json.dumps(record)


def test_provider_echo_cannot_persist_api_key_or_unapproved_headers() -> None:
    secret = "test-openai-secret"
    content = json.dumps(
        {
            "span_quality": "Good",
            "label_quality": "Good",
            "span_reason": f"Provider echoed {secret}.",
            "label_reason": "The description fits.",
        }
    )

    def transport(url, headers, payload, timeout):
        response = openai_response(content)
        return HTTPResponse(
            status=response.status,
            body=response.body,
            headers={
                "Content-Type": "application/json",
                "OpenAI-Request-ID": f"request-{secret}",
                "Authorization": f"Bearer {secret}",
                "Set-Cookie": f"credential={secret}",
                "X-Echo-Key": secret,
            },
        )

    record = run_openai_acceptability(
        acceptability_job(),
        api_key=secret,
        transport=transport,
    )

    serialized = json.dumps(record)
    assert secret not in serialized
    assert record["response"]["headers"] == {
        "content-type": "application/json",
        "openai-request-id": "request-[REDACTED]",
    }
    assert record["provider_content"] is not None
    assert REDACTED_SECRET in record["provider_content"]
    assert set(record["response"]["headers"]) <= SAFE_RESPONSE_HEADERS


def test_http_error_body_and_headers_are_secret_free() -> None:
    secret = "test-google-secret"

    def transport(url, headers, payload, timeout):
        return HTTPResponse(
            status=429,
            body=json.dumps({"error": f"quota failure for {secret}"}),
            headers={
                "Retry-After": "2",
                "X-Goog-Request-ID": f"request-{secret}",
                "X-Goog-Api-Key": secret,
            },
        )

    record = run_google_acceptability(
        acceptability_job(),
        api_key=secret,
        transport=transport,
    )

    assert secret not in json.dumps(record)
    assert record["raw_response"] == '{"error": "quota failure for [REDACTED]"}'
    assert record["response"]["headers"] == {
        "retry-after": "2",
        "x-goog-request-id": "request-[REDACTED]",
    }


def test_teacher_taxonomy_rejects_a_replaced_model_label(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.md"
    taxonomy = load_teacher_taxonomy()
    taxonomy_path.write_text(
        taxonomy.replace("  * Agreement —", "  * UnknownLabel —", 1),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="labels differ from the released label mapping"):
        load_teacher_taxonomy(taxonomy_path)
