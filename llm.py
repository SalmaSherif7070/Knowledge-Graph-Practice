import os
import json
import requests
from typing import Optional

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _get_api_keys() -> list[str]:
    keys = []
    primary = os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    i = 2
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    return keys


def call_gemini(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
    keys = _get_api_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY found in environment.")

    url_template = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={{key}}"

    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    last_error: Optional[Exception] = None

    for key in keys:
        try:
            resp = requests.post(
                url_template.format(key=key),
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code == 429:
                last_error = Exception(f"Quota exceeded for key ending ...{key[-4:]}")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.HTTPError as e:
            if resp.status_code == 429:
                last_error = e
                continue
            raise

    raise RuntimeError(f"All Gemini API keys exhausted or failed. Last error: {last_error}")


# ──────────────────────────────────────────────
# Conflict-specific prompt helpers
# ──────────────────────────────────────────────

CONFLICT_SYSTEM_PROMPT = """You are a strict rule conflict analyst.
Your job is to identify when two rules are in conflict.

Rules ARE in conflict if ANY of the following apply:
1. They cannot both be satisfied simultaneously.
2. Following one necessarily violates the other.
3. One rule creates an exception that directly undermines or negates the other rule's requirement.
4. They impose contradictory obligations on the same subject (e.g. must do X vs must not do X, or must do X vs may skip X).

Do NOT dismiss a conflict just because one rule frames itself as an "exception" or "emergency" provision.
If Rule B allows bypassing a requirement imposed by Rule A, that IS a conflict.

Respond ONLY in valid JSON."""

CONFLICT_PROMPT_TEMPLATE = """Analyze whether these two rules conflict:

Rule A (ID: {id_a}):
{text_a}

Rule B (ID: {id_b}):
{text_b}

A conflict exists if one rule imposes a requirement that the other rule contradicts, bypasses, or negates — even partially or in specific circumstances.

Respond with a JSON object:
{{
  "conflicts": true or false,
  "explanation": "brief explanation of the conflict or why there is none"
}}"""


def check_conflict_with_llm(
    id_a: str, text_a: str, id_b: str, text_b: str
) -> tuple[bool, str]:
    """Returns (conflicts: bool, explanation: str)."""
    prompt = CONFLICT_PROMPT_TEMPLATE.format(
        id_a=id_a, text_a=text_a, id_b=id_b, text_b=text_b
    )
    raw = call_gemini(prompt, system_prompt=CONFLICT_SYSTEM_PROMPT)

    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    clean = clean.removesuffix("```").strip()

    brace_start = clean.find("{")
    brace_end   = clean.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        clean = clean[brace_start : brace_end + 1]

    try:
        result = json.loads(clean)
        return bool(result.get("conflicts", False)), result.get("explanation", "")
    except json.JSONDecodeError:
        lower = raw.lower()
        conflicts = '"conflicts": true' in lower or (
            "conflict" in lower and '"conflicts": false' not in lower
        )
        return conflicts, raw