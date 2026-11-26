"""Single-side judge prompt for Exp L: span quality + label quality.

For one (sentence, edit_span, replacement, system_label, language), the judge
returns Good/Acceptable/Bad on each of two axes:

  span_quality  — is this a reasonable boundary for the underlying error?
                  Good       = exact, minimal span covering the error
                  Acceptable = covers the error but slightly too wide / narrow
                  Bad        = misses the error or includes unrelated tokens

  label_quality — is the system's label appropriate for the edit?
                  Good       = label is the most precise correct description
                  Acceptable = label is a coarser super-type that still applies
                  Bad        = label is wrong, irrelevant, or contradicts the edit

The prompt is one-shot and returns strict JSON. Reasoning is requested separately
for each axis to make the model commit to per-axis evidence rather than averaging.
"""

JUDGE_PROMPT = """You are a senior linguist evaluating one error-correction edit.

Language: {language}
Source sentence: "{source}"
Corrected sentence: "{target}"
The edit being judged: {edit_diff}
The classifier described this edit as: {label_gloss}

Rate the classifier on two independent axes. For each, choose ONE of:
  Good        — fully appropriate
  Acceptable  — defensible but not the best choice (span is not semantically clean, or a coarser super-type description)
  Bad         — wrong or misleading

span_quality:
  Good       = the span exactly covers the error, no more no less
  Acceptable = the span covers the error but is semantically confusing or not precise 
  Bad        = the span includes unrelated text that makes it unclear what error the edit is fixing

label_quality:
  Good       = the description is the most precise correct interpretation of this edit
  Acceptable = the description is a coarser but still valid interpretation
  Bad        = the description is wrong, irrelevant, or contradicts what the edit does

Return only a JSON object, no prose, no markdown:
{{
  "span_quality": "Good" | "Acceptable" | "Bad",
  "label_quality": "Good" | "Acceptable" | "Bad",
  "span_reason": "<one short sentence citing the source span>",
  "label_reason": "<one short sentence citing why the description fits or does not>"
}}
"""

JUDGE_SYSTEM_MESSAGE = "Return only valid JSON per the contract; no markdown, no extra text."
PRIMARY_JUDGE_MODEL = "GPT-5.4"
SECOND_JUDGE_MODEL = "Gemini 3.5 Flash"


def build_judge_messages(prompt_text: str) -> list[dict[str, str]]:
    """Build the exact system and user messages used by both paper judges."""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_text},
    ]


def build_judge_request(
    prompt_text: str,
    model: str,
    *,
    extra_body: dict | None = None,
) -> dict:
    """Build the OpenAI-compatible request payload used for evaluation."""

    payload = {
        "model": model,
        "messages": build_judge_messages(prompt_text),
        "temperature": 0,
    }
    if extra_body:
        payload["extra_body"] = extra_body
    else:
        payload.update(
            {
                "n": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "response_format": {"type": "json_object"},
            }
        )
    return payload


VALID = {"Good", "Acceptable", "Bad"}


def parse_response(text: str) -> dict:
    """Strict-ish JSON parser. Returns dict with the four expected keys, or empty values on failure."""
    import json

    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
        else:
            s = s.replace("```", "")
        if s.lower().lstrip().startswith("json"):
            s = s.lstrip()[4:].lstrip()
    s = s.strip()
    try:
        data = json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                data = json.loads(s[a : b + 1])
            except Exception:
                return _empty()
        else:
            return _empty()
    return {
        "span_quality": _coerce(data.get("span_quality")),
        "label_quality": _coerce(data.get("label_quality")),
        "span_reason": str(data.get("span_reason", "")),
        "label_reason": str(data.get("label_reason", "")),
    }


def _coerce(value):
    v = str(value).strip().capitalize() if value is not None else ""
    return v if v in VALID else ""


def _empty():
    return {"span_quality": "", "label_quality": "", "span_reason": "", "label_reason": ""}
