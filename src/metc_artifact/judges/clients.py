"""Direct public HTTP clients for fresh METC judge and teacher reruns."""

from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .protocols import (
    ACCEPTABILITY_SYSTEM_MESSAGE,
    QUALITY_LEVELS,
    TEACHER_SYSTEM_MESSAGE,
    ProtocolError,
    load_teacher_taxonomy,
    parse_acceptability_content,
    parse_teacher_content,
    render_acceptability_prompt,
    render_teacher_prompt,
    sha256_text,
    validate_teacher_taxonomy,
)


OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.4"
GOOGLE_MODEL = "gemini-3.5-flash"
GOOGLE_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"
)

SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "openai-request-id",
        "request-id",
        "retry-after",
        "x-goog-request-id",
        "x-request-id",
    }
)
REDACTED_SECRET = "[REDACTED]"

GOOGLE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "span_quality": {"type": "STRING", "enum": sorted(QUALITY_LEVELS)},
        "label_quality": {"type": "STRING", "enum": sorted(QUALITY_LEVELS)},
        "span_reason": {"type": "STRING"},
        "label_reason": {"type": "STRING"},
    },
    "required": ["span_quality", "label_quality", "span_reason", "label_reason"],
}


class RerunConfigurationError(RuntimeError):
    """Raised before a request when credentials or required inputs are missing."""


class TransportError(RuntimeError):
    """Raised when no HTTP response was received."""


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: str
    headers: Mapping[str, str]


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], HTTPResponse]


def urllib_json_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> HTTPResponse:
    """POST JSON with stdlib urllib; HTTP errors remain inspectable responses."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=int(response.status),
                body=response.read().decode("utf-8", errors="replace"),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as error:
        return HTTPResponse(
            status=int(error.code),
            body=error.read().decode("utf-8", errors="replace"),
            headers=dict(error.headers.items()) if error.headers else {},
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TransportError(str(error)) from error


def _api_key(explicit: str | None, environment_name: str) -> str:
    value = explicit or os.environ.get(environment_name)
    if not value:
        raise RerunConfigurationError(f"{environment_name} is required for this fresh rerun")
    return value


def _json_envelope(response: HTTPResponse) -> Mapping[str, Any]:
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise ProtocolError("provider HTTP response is not JSON") from error
    if not isinstance(value, Mapping):
        raise ProtocolError("provider HTTP response is not a JSON object")
    return value


def _redact_secret(value: str, secret: str) -> str:
    if not secret:
        return value
    return value.replace(secret, REDACTED_SECRET)


def _safe_response_headers(
    headers: Mapping[str, str],
    *,
    secret: str,
) -> dict[str, str]:
    """Retain only non-credential response metadata, with defense-in-depth redaction."""

    safe: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = str(raw_name).lower()
        if name in SAFE_RESPONSE_HEADERS:
            safe[name] = _redact_secret(str(raw_value), secret)
    return safe


def _openai_content(response: HTTPResponse) -> str:
    envelope = _json_envelope(response)
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ProtocolError("OpenAI response lacks choices[0].message.content") from error
    if not isinstance(content, str):
        raise ProtocolError("OpenAI message content is not text")
    return content


def _google_content(response: HTTPResponse) -> str:
    envelope = _json_envelope(response)
    try:
        parts = envelope["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as error:
        raise ProtocolError("Gemini response lacks candidates[0].content.parts") from error
    if not isinstance(parts, list):
        raise ProtocolError("Gemini content parts are not a list")
    texts = [part.get("text") for part in parts if isinstance(part, Mapping)]
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ProtocolError("Gemini response contains no text content")
    return "".join(texts)


def _base_record(
    *,
    run_kind: str,
    provider: str,
    model: str,
    stable_edit_ids: list[str],
    input_record: Mapping[str, Any],
    prompt: str,
    endpoint: str,
    configuration: Mapping[str, Any],
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "metc-fresh-rerun-v1",
        "source": "fresh_rerun",
        "authoritative_frozen_output": False,
        "run_kind": run_kind,
        "provider": provider,
        "model": model,
        "stable_edit_ids": list(stable_edit_ids),
        "input": copy.deepcopy(dict(input_record)),
        "prompt": {"text": prompt, "sha256": sha256_text(prompt)},
        "request": {
            "endpoint": endpoint,
            "configuration": copy.deepcopy(dict(configuration)),
            "payload": copy.deepcopy(dict(request_payload)),
        },
        "raw_response": None,
        "response": {"http_status": None, "headers": {}},
        "provider_content": None,
        "parsed_fields": None,
        "outputs_by_edit_id": {},
        "error": None,
    }


def _request_record(
    record: dict[str, Any],
    *,
    request_url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
    transport: Transport,
    content_reader: Callable[[HTTPResponse], str],
    content_parser: Callable[[str], dict[str, Any]],
    stable_output_builder: Callable[[dict[str, Any]], dict[str, dict[str, Any]]],
    secret: str,
) -> dict[str, Any]:
    try:
        response = transport(request_url, headers, payload, timeout)
    except Exception as error:  # transport injection may raise provider-specific errors
        message = _redact_secret(str(error), secret)
        record["error"] = {"type": "transport_error", "message": message, "status": None}
        return record

    safe_body = _redact_secret(response.body, secret)
    safe_response = HTTPResponse(
        status=response.status,
        body=safe_body,
        headers=_safe_response_headers(response.headers, secret=secret),
    )
    record["raw_response"] = safe_response.body
    record["response"] = {
        "http_status": safe_response.status,
        "headers": dict(safe_response.headers),
    }
    if not 200 <= safe_response.status < 300:
        record["error"] = {
            "type": "http_error",
            "message": f"provider returned HTTP {safe_response.status}",
            "status": safe_response.status,
        }
        return record
    try:
        content = content_reader(safe_response)
        record["provider_content"] = content
        parsed = content_parser(content)
        outputs = stable_output_builder(parsed)
    except (ProtocolError, KeyError, TypeError, ValueError) as error:
        record["error"] = {
            "type": "response_validation_error",
            "message": _redact_secret(str(error), secret),
            "status": safe_response.status,
        }
        return record
    record["parsed_fields"] = parsed
    record["outputs_by_edit_id"] = outputs
    return record


def _openai_acceptability_request(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = render_acceptability_prompt(job)
    edit_id = str(job["edit_id"])
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": ACCEPTABILITY_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
        "service_tier": "flex",
    }
    configuration = {
        "historical_backend_id": "openai_direct_gpt54_flex",
        "temperature": 0,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
        "service_tier": "flex",
    }
    return (
        _base_record(
            run_kind="acceptability",
            provider="openai",
            model=OPENAI_MODEL,
            stable_edit_ids=[edit_id],
            input_record=job,
            prompt=prompt,
            endpoint=OPENAI_ENDPOINT,
            configuration=configuration,
            request_payload=payload,
        ),
        payload,
    )


def _google_acceptability_request(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = render_acceptability_prompt(job)
    edit_id = str(job["edit_id"])
    payload = {
        "systemInstruction": {"parts": [{"text": ACCEPTABILITY_SYSTEM_MESSAGE}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": copy.deepcopy(GOOGLE_RESPONSE_SCHEMA),
        },
    }
    configuration = {
        "historical_backend_id": "vai_gemini35_flash",
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": copy.deepcopy(GOOGLE_RESPONSE_SCHEMA),
        "historical_thinking_configuration": {
            "status": "not_preserved",
            "fresh_request_behavior": "provider_default",
        },
    }
    return (
        _base_record(
            run_kind="acceptability",
            provider="google",
            model=GOOGLE_MODEL,
            stable_edit_ids=[edit_id],
            input_record=job,
            prompt=prompt,
            endpoint=GOOGLE_ENDPOINT,
            configuration=configuration,
            request_payload=payload,
        ),
        payload,
    )


def _openai_teacher_request(
    job: Mapping[str, Any],
    *,
    taxonomy_path: Path | None,
    taxonomy: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    if taxonomy is None:
        taxonomy = load_teacher_taxonomy(taxonomy_path)
    else:
        validate_teacher_taxonomy(taxonomy)
    prompt, edits = render_teacher_prompt(job, taxonomy=taxonomy)
    edit_ids = [str(edit["edit_id"]) for edit in edits]
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": "low",
        "service_tier": "flex",
    }
    configuration = {
        "historical_backend_id": "openai_direct_gpt54_flex",
        "reasoning_effort": "low",
        "service_tier": "flex",
        "temperature_explicitly_set": False,
        "response_format_explicitly_set": False,
        "taxonomy_sha256": sha256_text(taxonomy),
        "fixed_student_spans": True,
    }
    normalized_input = copy.deepcopy(dict(job))
    normalized_input["edits"] = copy.deepcopy(edits)
    record = _base_record(
        run_kind="annotation_teacher",
        provider="openai",
        model=OPENAI_MODEL,
        stable_edit_ids=edit_ids,
        input_record=normalized_input,
        prompt=prompt,
        endpoint=OPENAI_ENDPOINT,
        configuration=configuration,
        request_payload=payload,
    )
    return record, payload, edits, taxonomy


def build_request_contract(
    job: Mapping[str, Any],
    *,
    run_kind: str,
    provider: str,
    taxonomy_path: Path | None = None,
    taxonomy: str | None = None,
) -> dict[str, Any]:
    """Return the complete secret-free request contract used for resume identity."""

    if run_kind == "acceptability" and provider == "openai":
        record, _ = _openai_acceptability_request(job)
    elif run_kind == "acceptability" and provider == "google":
        record, _ = _google_acceptability_request(job)
    elif run_kind == "annotation_teacher" and provider == "openai":
        record, _, _, _ = _openai_teacher_request(
            job,
            taxonomy_path=taxonomy_path,
            taxonomy=taxonomy,
        )
    else:
        raise ValueError(f"unsupported request contract: {run_kind}/{provider}")
    return {
        key: copy.deepcopy(record[key])
        for key in (
            "run_kind",
            "provider",
            "model",
            "stable_edit_ids",
            "input",
            "prompt",
            "request",
        )
    }


def run_openai_acceptability(
    job: Mapping[str, Any],
    *,
    api_key: str | None = None,
    transport: Transport = urllib_json_transport,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Fresh primary-judge rerun using the public GPT-5.4 endpoint."""

    key = _api_key(api_key, "OPENAI_API_KEY")
    record, payload = _openai_acceptability_request(job)
    edit_id = str(job["edit_id"])
    return _request_record(
        record,
        request_url=OPENAI_ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        transport=transport,
        content_reader=_openai_content,
        content_parser=parse_acceptability_content,
        stable_output_builder=lambda parsed: {edit_id: dict(parsed)},
        secret=key,
    )


def run_google_acceptability(
    job: Mapping[str, Any],
    *,
    api_key: str | None = None,
    transport: Transport = urllib_json_transport,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Fresh second-judge rerun using Gemini's native generateContent API."""

    key = _api_key(api_key, "GOOGLE_API_KEY")
    record, payload = _google_acceptability_request(job)
    edit_id = str(job["edit_id"])
    return _request_record(
        record,
        request_url=GOOGLE_ENDPOINT,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        payload=payload,
        timeout=timeout,
        transport=transport,
        content_reader=_google_content,
        content_parser=parse_acceptability_content,
        stable_output_builder=lambda parsed: {edit_id: dict(parsed)},
        secret=key,
    )


def run_openai_teacher(
    job: Mapping[str, Any],
    *,
    api_key: str | None = None,
    taxonomy_path: Path | None = None,
    taxonomy: str | None = None,
    transport: Transport = urllib_json_transport,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Fresh annotation-teacher rerun over immutable caller-supplied student edits."""

    key = _api_key(api_key, "OPENAI_API_KEY")
    record, payload, edits, taxonomy = _openai_teacher_request(
        job,
        taxonomy_path=taxonomy_path,
        taxonomy=taxonomy,
    )
    return _request_record(
        record,
        request_url=OPENAI_ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
        transport=transport,
        content_reader=_openai_content,
        content_parser=lambda content: parse_teacher_content(content, edits, taxonomy=taxonomy),
        stable_output_builder=lambda parsed: copy.deepcopy(parsed["outputs_by_edit_id"]),
        secret=key,
    )
