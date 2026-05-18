import os
import json
import requests
from typing import Optional

# Support multiple Gemini API keys — if the first is exhausted, fall back to the next.
# Set them in .env as GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, etc.

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _get_api_keys() -> list[str]:
    """Collect all GEMINI_API_KEY* variables from environment, in order."""
    keys = []
    # Primary key
    primary = os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    # Additional keys: GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    i = 2
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    return keys


def call_gemini(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
    """
    Call Gemini generateContent API.
    Automatically falls back to the next API key on quota/rate-limit errors (429).
    """
    keys = _get_api_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY found in environment.")

    url_template = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={{key}}"

    contents = []
    if system_prompt:
        # Gemini uses system_instruction at the top level
        pass  # handled below

    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024,
        }
    }

    if system_prompt:
        body["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }

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
                continue  # try next key

            resp.raise_for_status()
            data = resp.json()
            # Extract text from response
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except requests.HTTPError as e:
            if resp.status_code == 429:
                last_error = e
                continue
            raise

    raise RuntimeError(
        f"All Gemini API keys exhausted or failed. Last error: {last_error}"
    )


# ──────────────────────────────────────────────
# Conflict-specific prompt helpers
# ──────────────────────────────────────────────

CONFLICT_SYSTEM_PROMPT = """You are a rule compliance expert.
Your job is to determine whether two rules directly conflict with each other.
Rules conflict when they cannot both be satisfied simultaneously, 
or when following one necessarily violates the other.
Respond ONLY in valid JSON."""

CONFLICT_PROMPT_TEMPLATE = """Analyze whether these two rules conflict:

Rule A (ID: {id_a}):
{text_a}

Rule B (ID: {id_b}):
{text_b}

Respond with a JSON object:
{{
  "conflicts": true or false,
  "explanation": "brief explanation of the conflict or why there is none"
}}"""


def check_conflict_with_llm(
    id_a: str, text_a: str, id_b: str, text_b: str
) -> tuple[bool, str]:
    """
    Returns (conflicts: bool, explanation: str).
    """
    prompt = CONFLICT_PROMPT_TEMPLATE.format(
        id_a=id_a, text_a=text_a, id_b=id_b, text_b=text_b
    )
    raw = call_gemini(prompt, system_prompt=CONFLICT_SYSTEM_PROMPT)

    # Strip markdown code fences if present
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(clean)
        return bool(result.get("conflicts", False)), result.get("explanation", "")
    except json.JSONDecodeError:
        # Fallback: look for keywords
        lower = raw.lower()
        conflicts = "true" in lower or "conflict" in lower
        return conflicts, raw